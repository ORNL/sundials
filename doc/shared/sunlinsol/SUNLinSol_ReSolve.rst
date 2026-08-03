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

.. _SUNLinSol.ReSolve:

The SUNLinSol_ReSolve Module
======================================

The SUNLinearSolver_ReSolve implementation of the ``SUNLinearSolver`` class is
designed to be used with the SUNMATRIX_RESOLVE matrix, and a CPU or GPU-enabled
vector. The header file to include when using this module is
``sunlinsol/sunlinsol_resolve.h``. The installed library to link to is
``libsundials_sunlinsolresolve.lib`` where ``lib`` is typically ``.so`` for
shared libraries and ``.a`` for static libraries.

.. warning::

   The SUNLinearSolver_ReSolve module is experimental and subject to change.


SUNLinearSolver_ReSolve Description
---------------------------------------

The SUNLinearSolver_ReSolve implementation provides an interface to the SystemSolver class
in the `ReSolve <https://resolve.readthedocs.io/en/latest/sphinx/user_guide/index.html#solver-subroutines>`_
library. 

When using this class, it is necessary to fully setup a ReSolve ``SystemSolver`` object.

The Re::Solve SystemSolver object
-------------------------------------

In general, when setting up this object the user should:

* Create the appropriate ``LinAlgWorkspace`` object.
* Set the appropriate solver settings.

To create a ``SystemSolver`` object, the user must provide the following,

* A ReSolve ``LinAlgWorkspace`` object (CPU, CUDA or HIP)
* A factorization method
* A refactorization method
* A preconditioner
* An iterative method

Below is an example of setting up the ``SystemSolver`` object with the following settings using CUDA:

* A CUDA workspace
* KLU factorization
* ``cusolverrf`` refactorization
* No preconditioner
* Flexible GMRES iterative refinement

.. code-block:: cpp

  /* Set up the ReSolve workspace*/
  ReSolve::LinAlgWorkspaceCUDA workspace;
  workspace.initializeHandles();

  /* ReSolve solver instatiation */
  ReSolve::SystemSolver solver(&workspace,
                               "klu",    // factorization
                               "cusolverrf", // refactorization
                               "cusolverrf", // triangular solve
                               "none",   // preconditioner
                               "fgmres"); // iterative refinement

Following is another example of setting up the ``SystemSolver`` object with the following settings using HIP

* A HIP workspace
* KLU factorization
* ``rocsolverrf`` refactorization
* No preconditioner
* No iterative refinement

.. code-block:: cpp

  /* Set up the ReSolve workspace*/
  ReSolve::LinAlgWorkspaceHIP workspace;
  workspace.initializeHandles();

  /* ReSolve solver instatiation */
  ReSolve::SystemSolver solver(&workspace,
                               "klu",    // factorization
                               "rocsolverrf", // refactorization
                               "rocsolverrf", // triangular solve
                               "none",   // preconditioner
                               "none"); // iterative refinement

More information about setting up this object can be found in the `ReSolve Docs <https://resolve.readthedocs.io/en/latest/sphinx/user_guide/index.html#solver-subroutines>`_.


SUNLinearSolver_ReSolve Functions
-------------------------------------

The SUNLinearSolver_ReSolve module defines the following implementations of
linear solver operations listed in :numref:`SUNLinSol.API`:

* ``SUNLinSolGetType_ReSolve``
* ``SUNLinSolInitialize_ReSolve``
* ``SUNLinSolSetup_ReSolve``
* ``SUNLinSolSolve_ReSolve``
* ``SUNLinSolLastFlag_ReSolve``
* ``SUNLinSolFree_ReSolve``

In addition, the module provides the following user-callable routines:


.. cpp:function:: SUNLinearSolver SUNLinSol_ReSolve(ReSolve::SystemSolver* solver, SUNMatrix A, ReSolve::memory::MemorySpace memspace, SUNContext sunctx)

   This constructor function creates and allocates memory for a
   ``SUNLinearSolver`` object.

   **Arguments:**
      * *solver* -- a Re::Solve SystemSolver object
      * *A* -- a SUNMATRIX_RESOLVE matrix for checking compatibility with the
        solver.
      * *memspace* -- A Re::Solve variable used to choose between performing an operation on 
        either the HOST and DEVICE. Choose ``ReSolve::memory::HOST`` to use CPU devices or 
        ``ReSolve::memory::DEVICE`` to use NVIDIA or AMD GPUs.
      * *sunctx* -- the :c:type:`SUNContext` object (see :numref:`SUNDIALS.SUNContext`)

   **Return value:**
      If successful, a ``SUNLinearSolver`` object. If *A* is
      incompatible then this routine will return ``NULL``. 

SUNLinearSolver_ReSolve Content
-----------------------------------

The SUNLinearSolver_ReSolve module defines the object *content* field of a
``SUNLinearSolver`` to be the following structure:

.. code-block:: cpp

   struct _SUNLinearSolverContent_ReSolve 
   {
    int last_flag;
    ReSolve::SystemSolver* solver;
    sunbooleantype factorized;
    ReSolve::memory::MemorySpace memspace;
   };

These entries of the *content* field contain the following
information:

* ``last_flag`` - last error return flag from internal function
  evaluations

* ``solver`` - The Re::Solve ``SystemSolver`` object

* ``factorized`` - flag indicating whether the factorization
  has ever been performed

* ``memspace`` - The Re::Solve memory space

