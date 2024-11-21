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
      fn_map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides)

    Args:
    ----
        fn: Function mapping floats-to-floats to apply.

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
        # Allocate local arrays for indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Output tensor index.
        in_index = cuda.local.array(MAX_DIMS, numba.int32)  # Input tensor index.

        # Get the global thread index (unique to each thread in the CUDA grid)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Ensure that the thread index is within bounds
        if i < out_size:
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
    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ... )

    Args:
    ----
        fn: Function mapping two floats to a float to apply.

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
    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor.
        size (int): Length of the input array `a`.

    """
    BLOCK_DIM = 32  # Number of threads per block

    # Shared memory array for intermediate sums
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Global thread index
    i = cuda.blockIdx.x * THREADS_PER_BLOCK + cuda.threadIdx.x

    # Local thread index within the block
    local_i = cuda.threadIdx.x

    # Initialize shared memory to zero
    cache[local_i] = 0.0
    if i < size:
        cache[local_i] = a[i]
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if local_i < stride:
            if i + stride < size:
                cache[local_i] += cache[local_i + stride]
        # Synchronize after each step to ensure correctness
        cuda.syncthreads()
        stride //= 2  # Halve the stride each iteration

    # Write the result from the first thread of each block to the output array
    if local_i == 0:
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

    def _reduce(
        out: Storage,  # Output tensor storage.
        out_shape: Shape,  # Shape of the output tensor.
        out_strides: Strides,  # Strides for the output tensor indexing.
        out_size: int,  # Total number of elements in the output tensor.
        a_storage: Storage,  # Input tensor storage.
        a_shape: Shape,  # Shape of the input tensor.
        a_strides: Strides,  # Strides for the input tensor indexing.
        reduce_dim: int,  # Dimension along which reduction is performed.
        reduce_value: float,  # Initial value for the reduction.
    ) -> None:
        # Shared memory cache to store intermediate results of the reduction.
        BLOCK_DIM = 1024  # Number of threads per block
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)

        # Local thread index within the block
        pos = cuda.threadIdx.x

        # Compute the output position in the output tensor
        out_pos = cuda.blockIdx.x

        # Ensure the output position is within bounds
        if out_pos >= out_size:
            return

        # Allocate arrays for indexing
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert the output position to a multidimensional index
        to_index(out_pos, out_shape, out_index)

        # Initialize the cache with the reduction value
        cache[pos] = reduce_value
        reduce_size = a_shape[reduce_dim]

        # Load elements along the reduction dimension into shared memory
        if pos < reduce_size:
            for d in range(MAX_DIMS):
                a_index[d] = out_index[d]  # Copy all dimensions
            a_index[reduce_dim] = pos  # Update the reduction dimension
            a_pos = index_to_position(a_index, a_strides)
            cache[pos] = a_storage[a_pos]
        else:
            cache[pos] = reduce_value  # Zero-padding for out-of-bounds elements
        cuda.syncthreads()

        # Perform reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride and (pos + stride) < reduce_size:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2  # Halve the stride each iteration

        # Write the result from the first thread of the block to the output tensor
        if pos == 0:
            out_position = index_to_position(out_index, out_strides)
            out[out_position] = cache[0]  # Store the reduced result

    # Return the CUDA JIT-compiled reduction function
    return cuda.jit()(_reduce)  # type: ignore


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
    cache_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    cache_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices within the block
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Load elements into shared memory if within bounds
    if i < size and j < size:
        # Compute flat indices for `a` and `b`
        a_idx = i * size + j
        b_idx = i * size + j
        # Load data into shared memory
        cache_a[i, j] = a[a_idx]
        cache_b[i, j] = b[b_idx]

    # Synchronize threads to ensure shared memory is ready
    cuda.syncthreads()

    # Perform matrix multiplication if within bounds
    if i < size and j < size:
        temp = 0.0  # Accumulator for the dot product
        for k in range(size):
            temp += cache_a[i, k] * cache_b[k, j]
        # Compute the flat index for the output tensor
        out_idx = i * size + j
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
    out: Storage,  # Output tensor storage.
    out_shape: Shape,  # Shape of the output tensor.
    out_strides: Strides,  # Strides for the output tensor.
    out_size: int,  # Total number of elements in the output tensor.
    a_storage: Storage,  # Input tensor A storage.
    a_shape: Shape,  # Shape of input tensor A.
    a_strides: Strides,  # Strides for input tensor A.
    b_storage: Storage,  # Input tensor B storage.
    b_shape: Shape,  # Shape of input tensor B.
    b_strides: Strides,  # Strides for input tensor B.
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must first be moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

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
    # Calculate batch strides considering broadcasting
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Batch index from the z-dimension of the grid
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32  # Tile size
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Compute the row and column indices of the output matrix
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Local thread indices within the tile
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Correct Dimension Mapping
    M, N, K = (
        out_shape[-2],  # Rows of the output matrix
        out_shape[-1],  # Columns of the output matrix
        a_shape[-1],  # Shared dimension
    )

    temp = 0.0  # Accumulator for the dot product

    # Number of tiles along the K dimension
    num_tiles = (K + BLOCK_DIM - 1) // BLOCK_DIM

    # Loop over all tiles along the K dimension
    for tile_idx in range(num_tiles):
        # Calculate the column index for `a` and row index for `b`
        a_k = tile_idx * BLOCK_DIM + ty
        b_k = tile_idx * BLOCK_DIM + tx

        # Load data into shared memory for `a`
        if (i < M) and (a_k < K):
            a_pos = batch * a_batch_stride + i * a_strides[1] + a_k * a_strides[2]
            a_shared[tx, ty] = a_storage[a_pos]
        else:
            a_shared[tx, ty] = 0.0  # Zero-padding for out-of-bounds

        # Load data into shared memory for `b`
        if (b_k < K) and (j < N):
            b_pos = batch * b_batch_stride + b_k * b_strides[1] + j * b_strides[2]
            b_shared[tx, ty] = b_storage[b_pos]
        else:
            b_shared[tx, ty] = 0.0  # Zero-padding for out-of-bounds

        # Synchronize to ensure all data is loaded into shared memory
        cuda.syncthreads()

        # Perform the dot product for the current tile
        for k in range(BLOCK_DIM):
            if (tile_idx * BLOCK_DIM + k) < K:
                temp += a_shared[tx, k] * b_shared[k, ty]

        # Synchronize before loading the next tile
        cuda.syncthreads()

    # Write the accumulated result to the output tensor if within bounds
    if (i < M) and (j < N):
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_pos] = temp


# Compile the tensor matrix multiply function using CUDA JIT
tensor_matrix_multiply = cuda.jit()(_tensor_matrix_multiply)
