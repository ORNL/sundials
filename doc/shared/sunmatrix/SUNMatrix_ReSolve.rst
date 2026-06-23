..
   Programmer(s): Jeffery Zhang
   ----------------------------------------------------------------
   SUNDIALS Copyright Start
   Copyright (c) 2025-2026, Lawrence Livermore National Security,
   University of Maryland Baltimore County, and the SUNDIALS contributors.
   Copyright (c) 2013-2025, Lawrence Livermore National Security
   and Southern Methodist University.
   Copyright (c) 2002-2013, Lawrence Livermore National Security.
   All rights reserved.

   See the top-level LICENSE and NOTICE files for details.

   SPDX-License-Identifier: BSD-3-Clause
   SUNDIALS Copyright End
   ----------------------------------------------------------------

.. _SUNMatrix.ReSolve:

The SUNMATRIX_RESOLVE Module
======================================

The SUNMATRIX_RESOLVE module interfaces to the
`Re::Solve <https://resolve.readthedocs.io/en/latest/>`_ library of GPU-resident linear
solvers designed to run on NVIDIA and AMD GPUs as well as on CPU devices. The module stores
matrices in the *compressed-sparse-row* (CSR) sparse matrix format.

More general information about sparse matrices can be found in the SUNMatrix_Sparse documentation
described in :numref:`SUNMatrix.Sparse`.


The header file to include when using this module is ``sunmatrix/sunmatrix_resolve.h``. 
The installed library to link to is ``libsundials_sunmatrixresolve.lib`` where ``lib`` is 
typically ``.so`` for shared libraries and ``.a`` for static libraries.


.. _SUNMatrix.ReSolve.functions:

SUNMATRIX_RESOLVE Functions
-----------------------------------


The SUNMATRIX_RESOLVE module currently defines the following implementations 
of matrix operations listed in :numref:`SUNMatrix.Ops`.

* ``SUNMatGetID_ReSolve`` -- returns ``SUNMATRIX_RESOLVE``
* ``SUNMatClone_ReSolve``
* ``SUNMatDestroy_ReSolve``
* ``SUNMatZero_ReSolve``

In addition, the SUNMATRIX_RESOLVE module defines the following
implementation specific functions:

.. cpp:function:: SUNMatrix SUNMatrix_ReSolve(sunindextype M, sunindextype N, sunindextype NNZ, ReSolve::memory::MemorySpace memspace, SUNContext sunctx)

   This constructor function creates and allocates memory for an
   :math:`M \times N` SUNMATRIX_RESOLVE ``SUNMatrix``.

   **Arguments:**
      * *M* -- the number of matrix rows.
      * *N* -- the number of matrix columns.
      * *NNZ* -- the number of non-zeros.
      * *memspace* -- A Re::Solve variable used to choose between performing an operation on 
        either the HOST and DEVICE. Choose ``ReSolve::memory::HOST`` to use CPU devices or 
        ``ReSolve::memory::DEVICE`` to use NVIDIA or AMD GPUs.
      * *sunctx* -- the :c:type:`SUNContext` object (see :numref:`SUNDIALS.SUNContext`)

   **Return value:**
      If successful, a ``SUNMatrix`` object otherwise ``NULL``.

.. cpp:function:: sunindextype SUNMatrix_ReSolve_Rows(SUNMatrix A)

   This function returns the number of rows in the ``SUNMatrix`` object.

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.

   **Return value:**
      The number of rows in the ``SUNMatrix`` object.


.. cpp:function:: sunindextype SUNMatrix_ReSolve_Columns(SUNMatrix A)

   This function returns the number of columns in the ``SUNMatrix`` object.
   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.

   **Return value:**
      The number of columns in the ``SUNMatrix`` object.


.. cpp:function:: sunindextype SUNMatrix_ReSolve_NNZ(SUNMatrix A)

   This function returns the number of nonzeros in a ``SUNMatrix``
   object.

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.

   **Return value:**
      The number of nonzeros in the ``SUNMatrix`` object.


.. cpp:function:: sunrealtype* SUNMatrix_ReSolve_Data(SUNMatrix A, ReSolve::memory::MemorySpace memspace)

   This function returns a pointer to the nonzero entries of the ``SUNMatrix``
   object on either the HOST or the DEVICE.

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.
      * *memspace* -- Either ``ReSolve::memory::HOST`` or ``ReSolve::memory::DEVICE``

   **Return value:**
      A pointer to the array of the nonzero entries of the ``SUNMatrix`` object on either
      the HOST or DEVICE.

.. cpp:function:: sunindextype* SUNMatrix_ReSolve_IndexValues(SUNMatrix A, ReSolve::memory::MemorySpace memspace)

   This function returns a pointer to the array of column indices of the ``SUNMatrix``
   object on either the HOST or the DEVICE.

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.
      * *memspace* -- Either ``ReSolve::memory::HOST`` or ``ReSolve::memory::DEVICE``

   **Return value:**
      A pointer to the array of column indices of the ``SUNMatrix`` object on either
      the HOST or DEVICE.

.. cpp:function:: sunindextype* SUNMatrix_ReSolve_IndexPointers(SUNMatrix A, ReSolve::memory::MemorySpace memspace)

   This function returns a pointer to the array of row pointers of the ``SUNMatrix``
   object on either the HOST or the DEVICE.

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.
      * *memspace* -- Either ``ReSolve::memory::HOST`` or ``ReSolve::memory::DEVICE``

   **Return value:**
      A pointer to the array of row pointers of the ``SUNMatrix`` object on either
      the HOST or DEVICE.

.. cpp:function:: SUNErrCode SUNMatrix_ReSolve_SetUpdated(SUNMatrix A, ReSolve::memory::MemorySpace memspace)

   This function is used to set a Re::Solve memory space to "updated". The other data
   mirror is also automatically set as non-updated.
   This function should be used if matrix data is updated through access of the raw pointers
   as the matrix has no way of knowing which data is most recent otherwise.

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.
      * *memspace* -- Either ``ReSolve::memory::HOST`` or ``ReSolve::memory::DEVICE``

   **Return value:**
      ``SUN_SUCCESS`` if the update is successful.
    
.. cpp:function:: SUNErrCode SUNMatrix_ReSolve_SyncData(SUNMatrix A, ReSolve::memory::MemorySpace memspace)

   This function is used to sync data in the given memory space with the updated memory space.

   The memory space other than the the given memory space must be up-to-date. Otherwise,
   the function will return an error. 

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.
      * *memspace* -- Either ``ReSolve::memory::HOST`` or ``ReSolve::memory::DEVICE``

   **Return value:**
      * ``SUN_SUCCESS`` if the sync is successful.
      * ``SUN_ERR_ARG_INCOMPATIBLE`` if the given memory space is not updated.

.. cpp:function:: void SUNMatrix_ReSolve_Print(SUNMatrix A)

   This function can be used to print the ``SUNMatrix`` object.

   **Arguments:**
      * *A* -- a ``SUNMatrix`` object.
      * *memspace* -- Either ``ReSolve::memory::HOST`` or ``ReSolve::memory::DEVICE``