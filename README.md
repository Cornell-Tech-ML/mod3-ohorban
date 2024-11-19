# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Task3_1 and Task3_2
output from running `(.venv) andromeda@Alexs-Mac-mini mod3-ohorban % python3 project/parallel_check.py`:

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (162)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (162)
--------------------------------------------------------------------------------|loop #ID
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):    |
        N = len(out)                                                            |
        # No longer define in_index and out_index here                          |
                                                                                |
        for i in prange(N):-----------------------------------------------------| #2
            # Define index arrays inside the loop                               |
            in_index = np.zeros(len(in_shape), dtype=np.int32)------------------| #0
            out_index = np.zeros(len(out_shape), dtype=np.int32)----------------| #1
                                                                                |
            to_index(i, out_shape, out_index)                                   |
            broadcast_index(out_index, out_shape, in_shape, in_index)           |
                                                                                |
            x = in_storage[index_to_position(in_index, in_strides)]             |
            index = index_to_position(out_index, out_strides)                   |
            out[index] = fn(x)                                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #2) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#2).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (168) is hoisted out
 of the parallel loop labelled #2 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (169) is hoisted out
 of the parallel loop labelled #2 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (202)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (202)
--------------------------------------------------------------------|loop #ID
    def _zip(                                                       |
        out,                                                        |
        out_shape,                                                  |
        out_strides,                                                |
        a_storage,                                                  |
        a_shape,                                                    |
        a_strides,                                                  |
        b_storage,                                                  |
        b_shape,                                                    |
        b_strides,                                                  |
    ):                                                              |
        N = len(out)                                                |
        # No longer define index arrays here                        |
                                                                    |
        for i in prange(N):-----------------------------------------| #6
            # Define index arrays inside the loop                   |
            out_index = np.zeros(len(out_shape), dtype=np.int32)----| #3
            a_in = np.zeros(len(a_shape), dtype=np.int32)-----------| #4
            b_in = np.zeros(len(b_shape), dtype=np.int32)-----------| #5
                                                                    |
            to_index(i, out_shape, out_index)                       |
            index = index_to_position(out_index, out_strides)       |
                                                                    |
            broadcast_index(out_index, out_shape, a_shape, a_in)    |
            a = a_storage[index_to_position(a_in, a_strides)]       |
                                                                    |
            broadcast_index(out_index, out_shape, b_shape, b_in)    |
            b = b_storage[index_to_position(b_in, b_strides)]       |
                                                                    |
            out[index] = fn(a, b)                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #6, #3, #4, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--6 is a parallel loop
   +--3 --> rewritten as a serial loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (serial)
   +--4 (serial)
   +--5 (serial)



Parallel region 0 (loop #6) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (218) is hoisted out
 of the parallel loop labelled #6 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (219) is hoisted out
 of the parallel loop labelled #6 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_in = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (220) is hoisted out
 of the parallel loop labelled #6 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_in = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (257)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (257)
--------------------------------------------------------------------------------------------|loop #ID
    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):    |
        N = len(out)                                                                        |
        reduce_dim_size = a_shape[reduce_dim]                                               |
                                                                                            |
        for i in prange(N):-----------------------------------------------------------------| #9
            out_index = np.zeros(len(out_shape), dtype=np.int32)----------------------------| #7
            a_index = np.zeros(len(a_shape), dtype=np.int32)--------------------------------| #8
                                                                                            |
            to_index(i, out_shape, out_index)                                               |
            index = index_to_position(out_index, out_strides)                               |
                                                                                            |
            # Initialize the accumulation variable                                          |
            acc = out[index]                                                                |
                                                                                            |
            for j in range(reduce_dim_size):                                                |
                # Copy out_index into a_index                                               |
                for k in range(len(a_shape)):                                               |
                    a_index[k] = out_index[k]                                               |
                a_index[reduce_dim] = j                                                     |
                                                                                            |
                a_val = a_storage[index_to_position(a_index, a_strides)]                    |
                acc = fn(acc, a_val)                                                        |
                                                                                            |
            out[index] = acc                                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #9, #7, #8).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--9 is a parallel loop
   +--8 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (serial)
   +--7 (serial)



Parallel region 0 (loop #9) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#9).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (262) is hoisted out
 of the parallel loop labelled #9 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (263) is hoisted out
 of the parallel loop labelled #9 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (285)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/andromeda/Desktop/mod3-ohorban/minitorch/fast_ops.py (285)
----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                  |
    out: Storage,                                                                             |
    out_shape: Shape,                                                                         |
    out_strides: Strides,                                                                     |
    a_storage: Storage,                                                                       |
    a_shape: Shape,                                                                           |
    a_strides: Strides,                                                                       |
    b_storage: Storage,                                                                       |
    b_shape: Shape,                                                                           |
    b_strides: Strides,                                                                       |
) -> None:                                                                                    |
    """NUMBA tensor matrix multiply function.                                                 |
                                                                                              |
    Should work for any tensor shapes that broadcast as long as                               |
                                                                                              |
        assert a_shape[-1] == b_shape[-2]                                                     |
                                                                                              |
    Optimizations:                                                                            |
                                                                                              |
    * Outer loop in parallel                                                                  |
    * No index buffers or function calls                                                      |
    * Inner loop should have no global writes, 1 multiply.                                    |
                                                                                              |
    Args:                                                                                     |
        out (Storage): storage for `out` tensor                                               |
        out_shape (Shape): shape for `out` tensor                                             |
        out_strides (Strides): strides for `out` tensor                                       |
        a_storage (Storage): storage for `a` tensor                                           |
        a_shape (Shape): shape for `a` tensor                                                 |
        a_strides (Strides): strides for `a` tensor                                           |
        b_storage (Storage): storage for `b` tensor                                           |
        b_shape (Shape): shape for `b` tensor                                                 |
        b_strides (Strides): strides for `b` tensor                                           |
                                                                                              |
    Returns:                                                                                  |
        None : Fills in `out`                                                                 |
    """                                                                                       |
    # Extract batch strides                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                    |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                    |
                                                                                              |
    assert a_shape[-1] == b_shape[-2]                                                         |
                                                                                              |
    N = len(out)                                                                              |
                                                                                              |
    # Precompute shape and stride elements to avoid indexing overhead                         |
    out_dim0, out_dim1, out_dim2 = out_shape[0], out_shape[1], out_shape[2]                   |
    out_stride0, out_stride1, out_stride2 = out_strides[0], out_strides[1], out_strides[2]    |
    a_stride1, a_stride2 = a_strides[1], a_strides[2]                                         |
    b_stride1, b_stride2 = b_strides[1], b_strides[2]                                         |
                                                                                              |
    for i in prange(N):-----------------------------------------------------------------------| #10
        # Compute indices for out tensor                                                      |
        out_0 = i // (out_dim1 * out_dim2)                                                    |
        rem = i % (out_dim1 * out_dim2)                                                       |
        out_1 = rem // out_dim2                                                               |
        out_2 = rem % out_dim2                                                                |
                                                                                              |
        # Compute positions in out storage                                                    |
        out_pos = out_0 * out_stride0 + out_1 * out_stride1 + out_2 * out_stride2             |
                                                                                              |
        # Compute starting positions in a and b storages                                      |
        a_start = out_0 * a_batch_stride + out_1 * a_stride1                                  |
        b_start = out_0 * b_batch_stride + out_2 * b_stride2                                  |
                                                                                              |
        temp = 0.0                                                                            |
        for k in range(a_shape[-1]):                                                          |
            # Access elements from a and b                                                    |
            a_pos = a_start + k * a_stride2                                                   |
            b_pos = b_start + k * b_stride1                                                   |
            temp += a_storage[a_pos] * b_storage[b_pos]                                       |
                                                                                              |
        out[out_pos] = temp                                                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Task3_3
Tests passing on google colab after running `!pytest -m task3_3`:
```
======================================= test session starts ========================================
platform linux -- Python 3.10.12, pytest-8.3.2, pluggy-1.5.0
rootdir: /content
plugins: hypothesis-6.54.0, env-1.1.4, typeguard-4.4.1, anyio-3.7.1
collected 117 items / 60 deselected / 57 selected

mod3-ohorban/tests/test_tensor_general.py .................................................. [ 87%]
.......                                                                                      [100%]

========================================= warnings summary =========================================
mod3-ohorban/tests/test_tensor_general.py:27
  /content/mod3-ohorban/tests/test_tensor_general.py:27: PytestUnknownMarkWarning: Unknown pytest.mark.task3_1 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    backend_tests = [pytest.param("fast", marks=pytest.mark.task3_1)]

mod3-ohorban/tests/test_tensor_general.py:30
  /content/mod3-ohorban/tests/test_tensor_general.py:30: PytestUnknownMarkWarning: Unknown pytest.mark.task3_2 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    matmul_tests = [pytest.param("fast", marks=pytest.mark.task3_2)]

mod3-ohorban/tests/test_tensor_general.py:35
  /content/mod3-ohorban/tests/test_tensor_general.py:35: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    backend_tests.append(pytest.param("cuda", marks=pytest.mark.task3_3))

mod3-ohorban/tests/test_tensor_general.py:38
  /content/mod3-ohorban/tests/test_tensor_general.py:38: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    matmul_tests.append(pytest.param("cuda", marks=pytest.mark.task3_4))

mod3-ohorban/tests/test_tensor_general.py:134
  /content/mod3-ohorban/tests/test_tensor_general.py:134: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:143
  /content/mod3-ohorban/tests/test_tensor_general.py:143: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:152
  /content/mod3-ohorban/tests/test_tensor_general.py:152: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:161
  /content/mod3-ohorban/tests/test_tensor_general.py:161: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:170
  /content/mod3-ohorban/tests/test_tensor_general.py:170: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:179
  /content/mod3-ohorban/tests/test_tensor_general.py:179: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:189
  /content/mod3-ohorban/tests/test_tensor_general.py:189: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:204
  /content/mod3-ohorban/tests/test_tensor_general.py:204: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:219
  /content/mod3-ohorban/tests/test_tensor_general.py:219: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:236
  /content/mod3-ohorban/tests/test_tensor_general.py:236: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:254
  /content/mod3-ohorban/tests/test_tensor_general.py:254: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:279
  /content/mod3-ohorban/tests/test_tensor_general.py:279: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:342
  /content/mod3-ohorban/tests/test_tensor_general.py:342: PytestUnknownMarkWarning: Unknown pytest.mark.task3_2 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_2

mod3-ohorban/tests/test_tensor_general.py: 14 warnings
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py: 4266 warnings
  /usr/local/lib/python3.10/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py: 11 warnings
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_args[cuda-fn5]
mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn7]
mod3-ohorban/tests/test_tensor_general.py::test_two_grad[cuda-fn2]
mod3-ohorban/tests/test_tensor_general.py::test_two_grad[cuda-fn4]
mod3-ohorban/tests/test_tensor_general.py::test_sum_practice2
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 27 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 9 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_sum_practice_other_dims
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================== 57 passed, 60 deselected, 4322 warnings in 244.33s (0:04:04) ===================
```


# Task3_4
Tests passing on google colab after running `!pytest -m task3_4`:

```
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content
plugins: hypothesis-6.54.0, env-1.1.4
collected 117 items / 110 deselected / 7 selected

mod3-ohorban/tests/test_tensor_general.py .......                                            [100%]

========================================= warnings summary =========================================
mod3-ohorban/tests/test_tensor_general.py:27
  /content/mod3-ohorban/tests/test_tensor_general.py:27: PytestUnknownMarkWarning: Unknown pytest.mark.task3_1 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    backend_tests = [pytest.param("fast", marks=pytest.mark.task3_1)]

mod3-ohorban/tests/test_tensor_general.py:30
  /content/mod3-ohorban/tests/test_tensor_general.py:30: PytestUnknownMarkWarning: Unknown pytest.mark.task3_2 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    matmul_tests = [pytest.param("fast", marks=pytest.mark.task3_2)]

mod3-ohorban/tests/test_tensor_general.py:35
  /content/mod3-ohorban/tests/test_tensor_general.py:35: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    backend_tests.append(pytest.param("cuda", marks=pytest.mark.task3_3))

mod3-ohorban/tests/test_tensor_general.py:38
  /content/mod3-ohorban/tests/test_tensor_general.py:38: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    matmul_tests.append(pytest.param("cuda", marks=pytest.mark.task3_4))

mod3-ohorban/tests/test_tensor_general.py:134
  /content/mod3-ohorban/tests/test_tensor_general.py:134: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:143
  /content/mod3-ohorban/tests/test_tensor_general.py:143: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:152
  /content/mod3-ohorban/tests/test_tensor_general.py:152: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:161
  /content/mod3-ohorban/tests/test_tensor_general.py:161: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:170
  /content/mod3-ohorban/tests/test_tensor_general.py:170: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:179
  /content/mod3-ohorban/tests/test_tensor_general.py:179: PytestUnknownMarkWarning: Unknown pytest.mark.task3_3 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_3

mod3-ohorban/tests/test_tensor_general.py:189
  /content/mod3-ohorban/tests/test_tensor_general.py:189: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:204
  /content/mod3-ohorban/tests/test_tensor_general.py:204: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:219
  /content/mod3-ohorban/tests/test_tensor_general.py:219: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:236
  /content/mod3-ohorban/tests/test_tensor_general.py:236: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:254
  /content/mod3-ohorban/tests/test_tensor_general.py:254: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:279
  /content/mod3-ohorban/tests/test_tensor_general.py:279: PytestUnknownMarkWarning: Unknown pytest.mark.task3_4 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_4

mod3-ohorban/tests/test_tensor_general.py:342
  /content/mod3-ohorban/tests/test_tensor_general.py:342: PytestUnknownMarkWarning: Unknown pytest.mark.task3_2 - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.task3_2

mod3-ohorban/tests/test_tensor_general.py::test_mul_practice1
mod3-ohorban/tests/test_tensor_general.py::test_mul_practice3
mod3-ohorban/tests/test_tensor_general.py::test_mul_practice3
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py: 111 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_mul_practice4
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_mul_practice4
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_mul_practice5
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 5 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 36 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

mod3-ohorban/tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 48 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 7 passed, 110 deselected, 154 warnings in 11.45s =========================
```


after running `!python3 project/timing.py` in google colab:

```

```