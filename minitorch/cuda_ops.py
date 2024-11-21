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
) -> Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: Function mapping floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    @cuda.jit()
    def _map(
        out: Storage,          # Output tensor storage.
        out_shape: Shape,      # Shape of the output tensor.
        out_strides: Strides,  # Strides for output tensor indexing.
        out_size: int,         # Total number of elements in the output tensor.
        in_storage: Storage,   # Input tensor storage.
        in_shape: Shape,       # Shape of the input tensor.
        in_strides: Strides,   # Strides for input tensor indexing.
    ) -> None:
        # Get the global thread index (unique to each thread in the CUDA grid)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Ensure that the thread index is within bounds
        if i >= out_size:
            return

        # Allocate local arrays for indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Output tensor index.
        in_index = cuda.local.array(MAX_DIMS, numba.int32)   # Input tensor index.

        # Convert the linear index 'i' to a multidimensional index for the output tensor
        to_index(i, out_shape, out_index)

        # Handle broadcasting: adjust indices to match input tensor's shape
        broadcast_index(out_index, out_shape, in_shape, in_index)

        # Calculate the flat positions in the input and output storages
        out_pos = index_to_position(out_index, out_strides)
        in_pos = index_to_position(in_index, in_strides)

        # Apply the function 'fn' to the input value and store the result in the output tensor
        out[out_pos] = fn(in_storage[in_pos])

    # Return the CUDA JIT-compiled function
    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: Function mapping two floats to a float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    @cuda.jit()
    def _zip(
        out: Storage,          # Output tensor storage.
        out_shape: Shape,      # Shape of the output tensor.
        out_strides: Strides,  # Strides for output tensor indexing.
        out_size: int,         # Total number of elements in the output tensor.
        a_storage: Storage,    # Storage for the first input tensor.
        a_shape: Shape,        # Shape of the first input tensor.
        a_strides: Strides,    # Strides for the first input tensor.
        b_storage: Storage,    # Storage for the second input tensor.
        b_shape: Shape,        # Shape of the second input tensor.
        b_strides: Strides,    # Strides for the second input tensor.
    ) -> None:
        # Get the global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Ensure the thread index is within bounds
        if i >= out_size:
            return

        # Allocate local arrays for indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Output tensor index.
        a_index = cuda.local.array(MAX_DIMS, numba.int32)    # First input tensor index.
        b_index = cuda.local.array(MAX_DIMS, numba.int32)    # Second input tensor index.

        # Convert the linear index 'i' to a multidimensional index for the output tensor
        to_index(i, out_shape, out_index)

        # Handle broadcasting for both input tensors
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # Calculate the flat positions in the input and output storages
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)

        # Apply the binary function 'fn' to the input values and store the result
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    # Return the CUDA JIT-compiled function
    return _zip


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array `a` of length `size` and an output array `out` of size `n // blockDim`,
    this function sums up each blockDim values into an out cell.

    For example, given:
        [a_1, a_2, ..., a_{100}]

    The output would be:
        [a_1 + ... + a_{31}, a_{32} + ... + a_{64}, ..., ]

    Note: Each block must perform the sum using shared memory!

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor.
        size (int): Length of the input array `a`.

    """
    BLOCK_DIM = THREADS_PER_BLOCK  # Number of threads per block

    # Shared memory array for intermediate sums
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Global thread index
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Local thread index within the block
    tid = cuda.threadIdx.x

    # Each thread loads one element from global memory to shared memory
    if i < size:
        cache[tid] = a[i]
    else:
        cache[tid] = 0.0  # Zero-padding for out-of-bounds elements

    # Synchronize threads within the block
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            cache[tid] += cache[tid + stride]
        # Synchronize after each step to ensure correctness
        cuda.syncthreads()
        stride //= 2  # Halve the stride each iteration

    # Write the result from the first thread of each block to the output array
    if tid == 0:
        out[cuda.blockIdx.x] = cache[0]  # Store the sum of the block


# Compile the sum practice function using CUDA JIT
jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Sum practice function using CUDA kernels."""
    (size,) = a.shape  # Get the size of the input tensor
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    out = TensorData([0.0 for _ in range(blockspergrid)], (blockspergrid,))
    out.to_cuda_()
    # Launch the kernel
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: Reduction function mapping two floats to a float.

    Returns:
    -------
        Tensor reduce function.

    """

    @cuda.jit()
    def _reduce(
        out: Storage,           # Output tensor storage.
        out_shape: Shape,       # Shape of the output tensor.
        out_strides: Strides,   # Strides for the output tensor.
        out_size: int,          # Total number of elements in the output tensor.
        a_storage: Storage,     # Input tensor storage.
        a_shape: Shape,         # Shape of the input tensor.
        a_strides: Strides,     # Strides for the input tensor.
        reduce_dim: int,        # Dimension along which reduction is performed.
        reduce_value: float,    # Initial value for the reduction.
    ) -> None:
        # Shared memory cache to store intermediate results of the reduction.
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)

        # Global thread index
        tid = cuda.threadIdx.x

        # Compute the position in the output tensor
        i = cuda.blockIdx.x

        if i >= out_size:
            return

        # Allocate arrays for indexing
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert the output index to multidimensional index
        to_index(i, out_shape, out_index)

        # Each thread processes elements along the reduction dimension
        reduce_dim_size = a_shape[reduce_dim]
        step = BLOCK_DIM

        # Initialize the accumulator with the reduce value
        acc = reduce_value

        # Iterate over the reduction dimension with a stride equal to block size
        for s in range(tid, reduce_dim_size, step):
            # Prepare the index for the input tensor
            for d in range(len(out_shape)):
                a_index[d] = out_index[d]
            a_index[reduce_dim] = s  # Update the reduction dimension

            # Calculate the flat index for the input tensor
            a_pos = index_to_position(a_index, a_strides)

            # Accumulate the result using the reduction function
            acc = fn(acc, a_storage[a_pos])

        # Store the partial result in shared memory
        cache[tid] = acc

        # Synchronize threads within the block
        cuda.syncthreads()

        # Perform reduction within the block
        offset = BLOCK_DIM // 2
        while offset > 0:
            if tid < offset:
                cache[tid] = fn(cache[tid], cache[tid + offset])
            cuda.syncthreads()
            offset //= 2

        # First thread writes the result to the output tensor
        if tid == 0:
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = cache[0]

    # Return the CUDA JIT-compiled reduction function
    return _reduce


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square matrix multiplication kernel to prepare for matmul.

    Given storages `a` and `b` representing square matrices of shape [size, size],
    this function computes the matrix multiplication and stores the result in `out`.

    Requirements:

    * All data must first be moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Computes:

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    out[i, j] += a[i, k] * b[k, j]

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor `a`.
        b (Storage): Storage for the input tensor `b`.
        size (int): Size of the square matrices.

    """
    BLOCK_DIM = 32  # Maximum block dimension (size <= 32)

    # Shared memory arrays for input matrices `a` and `b`
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices within the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Load elements into shared memory if within bounds
    if tx < size and ty < size:
        # Compute flat indices for `a` and `b`
        a_idx = tx * size + ty
        b_idx = tx * size + ty
        # Load data into shared memory
        shared_a[tx, ty] = a[a_idx]
        shared_b[tx, ty] = b[b_idx]
    else:
        # Zero-padding for threads outside the bounds
        shared_a[tx, ty] = 0.0
        shared_b[tx, ty] = 0.0

    # Synchronize threads to ensure shared memory is ready
    cuda.syncthreads()

    # Perform matrix multiplication if within bounds
    if tx < size and ty < size:
        temp = 0.0  # Accumulator for the dot product
        for k in range(size):
            temp += shared_a[tx, k] * shared_b[k, ty]
        # Compute the flat index for the output tensor
        out_idx = tx * size + ty
        # Write the result to global memory
        out[out_idx] = temp


# Compile the matrix multiplication practice function using CUDA JIT
jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Matrix multiply practice function using CUDA kernels."""
    (size, _) = a.shape  # Get the size of the matrices (assumed square)
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = (1, 1)
    out = TensorData([0.0 for _ in range(size * size)], (size, size))
    out.to_cuda_()
    # Launch the kernel
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,          # Output tensor storage.
    out_shape: Shape,      # Shape of the output tensor.
    out_strides: Strides,  # Strides for the output tensor.
    out_size: int,         # Total number of elements in the output tensor.
    a_storage: Storage,    # Input tensor A storage.
    a_shape: Shape,        # Shape of input tensor A.
    a_strides: Strides,    # Strides for input tensor A.
    b_storage: Storage,    # Input tensor B storage.
    b_shape: Shape,        # Shape of input tensor B.
    b_strides: Strides,    # Strides for input tensor B.
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must first be moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as:

        assert a_shape[-1] == b_shape[-2]

    Args:
    ----
        out (Storage): Storage for the output tensor.
        out_shape (Shape): Shape of the output tensor.
        out_strides (Strides): Strides for the output tensor.
        out_size (int): Total number of elements in the output tensor.
        a_storage (Storage): Storage for tensor `a`.
        a_shape (Shape): Shape of tensor `a`.
        a_strides (Strides): Strides for tensor `a`.
        b_storage (Storage): Storage for tensor `b`.
        b_shape (Shape): Shape of tensor `b`.
        b_strides (Strides): Strides for tensor `b`.

    Returns:
    -------
        None: Fills in `out` with the result of matrix multiplication.

    """
    # Define the tile size
    TILE_SIZE = 32  # Assuming size <= 32 for shared memory limits

    # Allocate shared memory for tiles of `a` and `b`
    shared_a = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    shared_b = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    # Compute the batch index
    batch_index = cuda.blockIdx.z

    # Compute the row and column indices for `out`
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Local thread indices within the block
    local_row = cuda.threadIdx.y
    local_col = cuda.threadIdx.x

    # Initialize the accumulator for the dot product
    acc = 0.0

    # Adjust batch indices for broadcasting if applicable
    a_batch = batch_index if a_shape[0] > 1 else 0  # Batch index for `a`
    b_batch = batch_index if b_shape[0] > 1 else 0  # Batch index for `b`

    # Number of tiles along the K dimension (shared dimension)
    K = a_shape[-1]  # Shared dimension (columns of `a` or rows of `b`)
    num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE  # Total tiles required

    # Iterate over tiles
    for tile_idx in range(num_tiles):
        # Compute global indices for the current tile in `a`
        a_row = row
        a_col = tile_idx * TILE_SIZE + local_col

        # Compute global indices for the current tile in `b`
        b_row = tile_idx * TILE_SIZE + local_row
        b_col = col

        # Load data into shared memory for `a`
        if a_row < a_shape[-2] and a_col < a_shape[-1]:
            a_index = cuda.local.array(MAX_DIMS, numba.int32)  # Index array for `a`
            a_index[0] = a_batch
            a_index[1] = a_row
            a_index[2] = a_col
            a_pos = index_to_position(a_index, a_strides)
            shared_a[local_row, local_col] = a_storage[a_pos]
        else:
            shared_a[local_row, local_col] = 0.0  # Zero-padding for out-of-bounds

        # Load data into shared memory for `b`
        if b_row < b_shape[-2] and b_col < b_shape[-1]:
            b_index = cuda.local.array(MAX_DIMS, numba.int32)  # Index array for `b`
            b_index[0] = b_batch
            b_index[1] = b_row
            b_index[2] = b_col
            b_pos = index_to_position(b_index, b_strides)
            shared_b[local_row, local_col] = b_storage[b_pos]
        else:
            shared_b[local_row, local_col] = 0.0  # Zero-padding for out-of-bounds

        # Synchronize threads to ensure all data is loaded
        cuda.syncthreads()

        # Perform the computation on the tile
        for k in range(TILE_SIZE):
            acc += shared_a[local_row, k] * shared_b[k, local_col]

        # Synchronize before loading the next tile
        cuda.syncthreads()

    # Write the result to global memory if within bounds
    if row < out_shape[-2] and col < out_shape[-1]:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Index array for `out`
        out_index[0] = batch_index
        out_index[1] = row
        out_index[2] = col
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = acc


# Compile the tensor matrix multiply function using CUDA JIT
tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)

