# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to compile a function for the GPU."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to compile a function for the GPU."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """See `tensor_ops.py`"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,  # Output tensor storage.
        out_shape: Shape,  # Shape of the output tensor.
        out_strides: Strides,  # Strides for output tensor indexing.
        out_size: int,  # Total number of elements in the output tensor.
        in_storage: Storage,  # Input tensor storage.
        in_shape: Shape,  # Shape of the input tensor.
        in_strides: Strides,  # Strides for input tensor indexing.
    ) -> None:
        # Get the global thread index (unique to each thread in the CUDA grid)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Ensure that the thread index is within bounds
        if i >= out_size:
            return

        # Allocate local arrays for indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Output tensor index.
        in_index = cuda.local.array(MAX_DIMS, numba.int32)  # Input tensor index.

        # Convert the flat index i to a multi-dimensional index for the output tensor
        to_index(i, out_shape, out_index)

        # Broadcast the output index to match the input tensor's shape
        broadcast_index(out_index, out_shape, in_shape, in_index)

        # Calculate the flat positions in the input and output storages
        in_pos = index_to_position(in_index, in_strides)
        out_pos = index_to_position(out_index, out_strides)

        # Apply the function fn to the input value and store the result in the output tensor
        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(
        _map
    )  # Use CUDA JIT compilation to transform the function into a GPU kernel.


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,  # Output tensor storage.
        out_shape: Shape,  # Shape of the output tensor.
        out_strides: Strides,  # Strides for output tensor indexing.
        out_size: int,  # Total number of elements in the output tensor.
        a_storage: Storage,  # Storage for the first input tensor.
        a_shape: Shape,  # Shape of the first input tensor.
        a_strides: Strides,  # Strides for the first input tensor.
        b_storage: Storage,  # Storage for the second input tensor.
        b_shape: Shape,  # Shape of the second input tensor.
        b_strides: Strides,  # Strides for the second input tensor.
    ) -> None:
        # Get the global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Ensure the thread index is within bounds
        if i >= out_size:
            return

        # Allocate local arrays for indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Output tensor index.
        a_index = cuda.local.array(MAX_DIMS, numba.int32)  # First input tensor index.
        b_index = cuda.local.array(MAX_DIMS, numba.int32)  # Second input tensor index.

        # Convert the flat index i to a multi-dimensional index for the output tensor
        to_index(i, out_shape, out_index)

        # Compute corresponding indices for the input tensors, considering broadcasting
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # Calculate the flat positions in the input and output storages
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)

        # Apply the binary function fn to the input values and store the result
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(
        _zip
    )  # Use CUDA JIT compilation to enable parallel execution across GPU threads.


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # Shared memory array
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Global thread index
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Local thread index within the block
    tid = cuda.threadIdx.x

    # Each thread loads one element from global to shared memory
    if i < size:
        cache[tid] = a[i]
    else:
        cache[tid] = 0.0

    # Synchronize threads within the block
    cuda.syncthreads()

    # Perform reduction in shared memory
    offset = 1
    while offset < BLOCK_DIM:
        cuda.syncthreads()
        if tid % (2 * offset) == 0 and (tid + offset) < BLOCK_DIM:
            cache[tid] += cache[tid + offset]
        offset *= 2

    # Write result to output
    if tid == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = jit(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Sum practice function."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,  # Output tensor storage.
        out_shape: Shape,  # Shape of the output tensor.
        out_strides: Strides,  # Strides for the output tensor.
        out_size: int,  # Total number of elements in the output tensor.
        a_storage: Storage,  # Input tensor storage.
        a_shape: Shape,  # Shape of the input tensor.
        a_strides: Strides,  # Strides for the input tensor.
        reduce_dim: int,  # Dimension along which reduction is performed.
        reduce_value: float,  # Initial value for the reduction.
    ) -> None:
        BLOCK_DIM = 1024  # Maximum number of threads in a block.

        # Shared memory cache to store intermediate results of the reduction.
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)

        # Global block index corresponding to the current output element.
        out_idx = cuda.blockIdx.x

        # Local thread index within the block.
        tid = cuda.threadIdx.x

        # Number of dimensions in the output tensor.
        num_dims = len(out_shape)

        # Allocate temporary arrays for multi-dimensional indexing.
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert the global block index to a multi-dimensional index for the output tensor.
        to_index(out_idx, out_shape, out_index)

        # Initialize the shared memory cache for this thread to the reduction's initial value.
        cache[tid] = reduce_value

        # Size of the reduction dimension in the input tensor.
        reduce_dim_size = a_shape[reduce_dim]

        # Process multiple elements along the reduction dimension.
        idx = tid  # Start with the thread ID.
        while idx < reduce_dim_size:
            # Copy the current output index to the input index and update the reduce dimension.
            for d in range(num_dims):
                a_index[d] = out_index[d]
            a_index[reduce_dim] = idx

            # Calculate the flat position of the current input element in storage.
            a_pos = index_to_position(a_index, a_strides)

            # Apply the reduction function to accumulate the result in shared memory.
            cache[tid] = fn(cache[tid], a_storage[a_pos])

            # Move to the next element along the reduction dimension.
            idx += BLOCK_DIM

        # Synchronize threads within the block to ensure all threads have finished processing.
        cuda.syncthreads()

        # Perform the reduction within the block using a binary tree structure.
        offset = BLOCK_DIM // 2
        while offset > 0:
            if tid < offset:
                # Combine the results from two halves of the shared memory cache.
                cache[tid] = fn(cache[tid], cache[tid + offset])
            # Synchronize after each step to ensure cache consistency.
            cuda.syncthreads()
            offset //= 2  # Halve the offset to progressively reduce the data.

        # The first thread writes the final reduced value for this block to the output tensor.
        if tid == 0:
            # Calculate the storage position of the current output element.
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = cache[0]  # Write the reduced value to the output tensor.

    # Compile the reduction function into a CUDA kernel for parallel execution.
    return cuda.jit()(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32  # Maximum block dimension. All matrices have size <= 32.

    # Shared memory buffers for the input matrices a and b.
    shared_a = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices in the block (y: row, x: column).
    y = numba.cuda.threadIdx.y
    x = numba.cuda.threadIdx.x

    # Load data into shared memory if within bounds.
    if x < size and y < size:
        shared_a[y, x] = a[y * size + x]  # Load element of a into shared memory.
        shared_b[y, x] = b[y * size + x]  # Load element of b into shared memory.
    else:
        shared_a[y, x] = 0  # Zero-pad out-of-bounds elements.
        shared_b[y, x] = 0
    numba.cuda.syncthreads()  # Synchronize all threads to ensure shared memory is ready.

    # Perform matrix multiplication if the thread is within matrix bounds.
    if y < size and x < size:
        temp = 0  # Temporary accumulator for the result.
        for val in range(size):
            # Compute dot product for the row of a and column of b.
            temp += shared_a[y, val] * shared_b[val, x]
        # Write the final result to global memory.
        out[y * size + x] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Matrix multiply practice function."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Initialize the accumulator for the dot product.
    temp = 0.0

    # Number of tiles along the shared dimension (K dimension).
    K = a_shape[-1]  # Shared dimension (columns of a or rows of b).
    num_tiles = (K + BLOCK_DIM - 1) // BLOCK_DIM  # Total tiles required.

    # Adjust batch indices for broadcasting if applicable.
    a_batch = batch if a_shape[0] > 1 else 0  # Batch index for a.
    b_batch = batch if b_shape[0] > 1 else 0  # Batch index for b.

    for tile_idx in range(num_tiles):
        # Compute global indices for the current tile in a and b.
        a_i = i
        a_j = tile_idx * BLOCK_DIM + pj  # Column index for the tile in a.

        b_i = tile_idx * BLOCK_DIM + pi  # Row index for the tile in b.
        b_j = j

        # Load data from global memory to shared memory for a, with boundary checks.
        if a_i < a_shape[-2] and a_j < a_shape[-1]:
            a_index = cuda.local.array(MAX_DIMS, numba.int32)  # Index array for a.
            a_index[0] = a_batch
            a_index[1] = a_i
            a_index[2] = a_j
            a_pos = index_to_position(a_index, a_strides)  # Flat index for a.
            a_shared[pi, pj] = a_storage[a_pos]  # Load into shared memory.
        else:
            a_shared[pi, pj] = 0.0  # Zero-padding for out-of-bounds elements.

        # Load data from global memory to shared memory for b, with boundary checks.
        if b_i < b_shape[-2] and b_j < b_shape[-1]:
            b_index = cuda.local.array(MAX_DIMS, numba.int32)  # Index array for b.
            b_index[0] = b_batch
            b_index[1] = b_i
            b_index[2] = b_j
            b_pos = index_to_position(b_index, b_strides)  # Flat index for b.
            b_shared[pi, pj] = b_storage[b_pos]  # Load into shared memory.
        else:
            b_shared[pi, pj] = 0.0  # Zero-padding for out-of-bounds elements.

        # Synchronize threads to ensure all data is loaded into shared memory.
        cuda.syncthreads()

        # Compute the partial dot product for the tile.
        for k in range(BLOCK_DIM):
            temp += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize before moving to the next tile.
        cuda.syncthreads()

    # Write the accumulated result to global memory if within bounds.
    if i < out_shape[-2] and j < out_shape[-1]:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Index array for out.
        out_index[0] = batch
        out_index[1] = i
        out_index[2] = j
        out_pos = index_to_position(out_index, out_strides)  # Flat index for out.
        out[out_pos] = temp  # Store the result in global memory.


# Compile the function into a CUDA kernel for use.
tensor_matrix_multiply = jit(_tensor_matrix_multiply)
