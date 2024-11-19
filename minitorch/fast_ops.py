from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile a function with NUMBA."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
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

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See tensor_ops.py for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

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
        in_storage: Storage,  # Input tensor storage.
        in_shape: Shape,  # Shape of the input tensor.
        in_strides: Strides,  # Strides for input tensor indexing.
    ) -> None:
        """Apply the function `fn` to each element of the input tensor and store in the output."""
        N = len(out)  # Total number of elements in the output tensor.
        for i in prange(N):  # Iterate in parallel over all elements.
            # Temporary arrays to hold indices for input and output tensors.
            in_index = np.zeros(len(in_shape), dtype=np.int32)
            out_index = np.zeros(len(out_shape), dtype=np.int32)

            # Convert linear index i to multidimensional indices for the output tensor.
            to_index(i, out_shape, out_index)

            # Broadcast the output index to match the input tensor shape.
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Access the corresponding element in the input tensor using strides.
            x = in_storage[index_to_position(in_index, in_strides)]

            # Calculate the position in the output tensor storage.
            index = index_to_position(out_index, out_strides)

            # Apply the function fn and store the result in the output tensor.
            out[index] = fn(x)

    return njit(
        _map, parallel=True
    )  # Use Numba to compile the function with parallelism enabled.


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See tensor_ops.py for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,  # Output tensor.
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,  # First input tensor.
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,  # Second input tensor.
    ) -> None:
        """Apply the binary function `fn` to elements from two tensors."""
        N = len(out)  # Total elements in the output tensor.
        for i in prange(N):  # Parallel iteration over elements.
            # Temporary index arrays for output, and the two input tensors.
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_in = np.zeros(len(a_shape), dtype=np.int32)
            b_in = np.zeros(len(b_shape), dtype=np.int32)

            # Convert linear index i to multidimensional indices for the output tensor.
            to_index(i, out_shape, out_index)

            # Map the output indices to positions in the two input tensors via broadcasting.
            broadcast_index(out_index, out_shape, a_shape, a_in)
            broadcast_index(out_index, out_shape, b_shape, b_in)

            # Access elements in the input tensors using strides.
            a = a_storage[index_to_position(a_in, a_strides)]
            b = b_storage[index_to_position(b_in, b_strides)]

            # Store the result of applying fn in the output tensor.
            out[index_to_position(out_index, out_strides)] = fn(a, b)

    return njit(
        _zip, parallel=True
    )  # Use Numba to compile the function with parallelism enabled.


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See tensor_ops.py for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,  # Output tensor.
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,  # Input tensor.
        reduce_dim: int,  # Dimension along which reduction is performed.
    ) -> None:
        """Perform reduction along the specified dimension."""
        N = len(out)  # Total elements in the output tensor.
        reduce_dim_size = a_shape[reduce_dim]  # Size of the dimension being reduced.

        for i in prange(N):  # Parallel iteration over output elements.
            # Temporary arrays for indexing input and output tensors.
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)

            # Convert linear index i to multidimensional index for the output tensor.
            to_index(i, out_shape, out_index)

            # Calculate the position in the output tensor storage.
            out_pos = index_to_position(out_index, out_strides)

            # Initialize the accumulator with the first element for reduction.
            acc = out[out_pos]

            for j in range(reduce_dim_size):  # Iterate over the reduction dimension.
                # Copy out_index to a_index and modify the reduce dimension.
                a_index[:] = out_index
                a_index[reduce_dim] = j

                # Fetch the value to reduce and update the accumulator.
                a_val = a_storage[index_to_position(a_index, a_strides)]
                acc = fn(acc, a_val)

            # Store the result of the reduction in the output tensor.
            out[out_pos] = acc

    return njit(
        _reduce, parallel=True
    )  # Use Numba to compile the function with parallelism enabled.


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

        assert a_shape[-1] == b_shape[-2]

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.

    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # Extract batch strides
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Calculate batch stride offsets only if the batch dimension > 1.
    # If batch size is 1, the tensors are not batched, so the stride is 0.

    assert a_shape[-1] == b_shape[-2]
    # Ensure that the inner dimensions of the matrices are compatible for multiplication:
    # The number of columns in a must equal the number of rows in b.

    N = len(out)  # Total number of elements in the output tensor.

    # Precompute shape and stride elements to reduce overhead during iteration
    _, out_dim1, out_dim2 = out_shape[0], out_shape[1], out_shape[2]
    out_stride0, out_stride1, out_stride2 = (
        out_strides[0],
        out_strides[1],
        out_strides[2],
    )
    a_stride1, a_stride2 = a_strides[1], a_strides[2]
    b_stride1, b_stride2 = b_strides[1], b_strides[2]

    for i in prange(N):
        # Iterate in parallel over all elements of the output tensor.

        # Compute the 3D indices for the current output tensor element.
        out_0 = i // (out_dim1 * out_dim2)
        rem = i % (out_dim1 * out_dim2)
        out_1 = rem // out_dim2
        out_2 = rem % out_dim2

        # Compute the position of the current element in the output storage.
        out_pos = out_0 * out_stride0 + out_1 * out_stride1 + out_2 * out_stride2

        # Compute starting positions in the input tensors a and b.
        a_start = out_0 * a_batch_stride + out_1 * a_stride1
        b_start = out_0 * b_batch_stride + out_2 * b_stride2

        # Perform the dot product for the current output element.
        temp = 0.0  # Temporary accumulator for the result of the dot product.
        for k in range(a_shape[-1]):
            # Iterate over the shared dimension of a and b to compute the dot product.

            # Calculate the position of the current element in a and b.
            a_pos = a_start + k * a_stride2  # Access row from a.
            b_pos = b_start + k * b_stride1  # Access column from b.

            # Multiply and accumulate the corresponding elements.
            temp += a_storage[a_pos] * b_storage[b_pos]

        # Store the result of the dot product in the output tensor.
        out[out_pos] = temp


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
