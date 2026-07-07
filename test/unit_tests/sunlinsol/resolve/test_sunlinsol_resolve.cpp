/*
 * -----------------------------------------------------------------
 * Programmer(s): Jeffery Zhang
 * 
 * Based off the test test_sunlinsol_klu.c written by Daniel Reynolds @UMBC
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2025-2026, Lawrence Livermore National Security,
 * University of Maryland Baltimore County, and the SUNDIALS contributors.
 * Copyright (c) 2013-2025, Lawrence Livermore National Security
 * and Southern Methodist University.
 * Copyright (c) 2002-2013, Lawrence Livermore National Security.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the testing routine to check the SUNLinSol ReSolve module
 * implementation.
 * -----------------------------------------------------------------
 */

#include <nvector/nvector_serial.h>
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunmatrix/sunmatrix_resolve.hpp>
#include <sunlinsol/sunlinsol_resolve.hpp>

// ReSolve headers
#include <resolve/SystemSolver.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include "test_sunlinsol.h"

/* ----------------------------------------------------------------------
 * SUNLinSol_ReSolve Linear Solver Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  int fails = 0;      /* counter for test failures  */
  sunindextype N;     /* matrix columns, rows       */
  SUNLinearSolver LS; /* linear solver object       */
  SUNMatrix A, B, C;     /* test matrices              */
  N_Vector x, y, b;   /* test vectors               */
  sunrealtype *matdata, *xdata;
  int print_timing;
  sunindextype i, j, k;
  SUNContext sunctx;

  if (SUNContext_Create(SUN_COMM_NULL, &sunctx))
  {
    printf("ERROR: SUNContext_Create failed\n");
    return (-1);
  }

  /* check input and set matrix dimensions */
  if (argc < 3)
  {
    printf("ERROR: TWO (2) Inputs required: matrix size, "
           "print timing \n");
    return (-1);
  }

  N = (sunindextype)atoi(argv[1]);
  if (N <= 0)
  {
    printf("ERROR: matrix size must be a positive integer \n");
    return (-1);
  }

  print_timing = atoi(argv[2]);
  SetTiming(print_timing);

  printf("\nReSolve linear solver test: size %ld, \n\n", (long int)N);

  /* Create matrices and vectors */
  C = SUNDenseMatrix(N, N, sunctx);
  x = N_VNew_Serial(N, sunctx);
  y = N_VNew_Serial(N, sunctx);
  b = N_VNew_Serial(N, sunctx);

  /* Fill matrix with uniform random data in [0,1/N] */
  for (k = 0; k < 5 * N; k++)
  {
    i          = rand() % N;
    j          = rand() % N;
    matdata    = SUNDenseMatrix_Column(C, j);
    matdata[i] = (sunrealtype)rand() / (sunrealtype)RAND_MAX / N;
  }

  /* Add identity to matrix */
  fails = SUNMatScaleAddI(ONE, C);
  if (fails)
  {
    printf("FAIL: SUNLinSol SUNMatScaleAddI failure\n");
    return (1);
  }

  /* Fill x vector with uniform random data in [0,1] */
  xdata = N_VGetArrayPointer(x);
  for (i = 0; i < N; i++)
  {
    xdata[i] = (sunrealtype)rand() / (sunrealtype)RAND_MAX;
  }

  /* Create sparse matrix from dense, and destroy B */
  B = SUNSparseFromDenseMatrix(C, ZERO, 1);
  SUNMatDestroy(C);

  /* copy x into y to print in case of solver failure */
  N_VScale(ONE, x, y);

  /* create right-hand side vector for linear solve */
  fails = SUNMatMatvec(B, x, b);
  if (fails)
  {
    printf("FAIL: SUNLinSol SUNMatMatvec failure\n");
    return (1);
  }

  /* Create ReSolve SUNMatrix from the Sparse Matrix */

  // Initialize a ReSolve HOST memory space.
  ReSolve::memory::MemorySpace memspace = ReSolve::memory::HOST;
  std::string hwbackend = "CPU";
// Check if a GPU backend is enabled
#ifdef SUNDIALS_RESOLVE_BACKENDS_CUDA
  hwbackend = "CUDA";
  memspace  = ReSolve::memory::DEVICE;
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  hwbackend = "HIP";
  memspace  = ReSolve::memory::DEVICE;
#endif

  // Create a SUNMatrix_ReSolve object
  A = SUNMatrix_ReSolve(N, N,  SUNSparseMatrix_NNZ(B), memspace, sunctx);

  // Get pointers to content arrays
  sunrealtype* data = SUNMatrix_ReSolve_Data(A, ReSolve::memory::HOST);
  sunindextype* index_values =
    SUNMatrix_ReSolve_IndexValues(A, ReSolve::memory::HOST);
  sunindextype* index_pointers =
    SUNMatrix_ReSolve_IndexPointers(A, ReSolve::memory::HOST);

  // Copy the data in the Sparse matrix pointers to the ReSolve pointers
  memcpy(data, SUNSparseMatrix_Data(B), SUNSparseMatrix_NNZ(B) * sizeof(sunrealtype));

  memcpy(index_values, SUNSparseMatrix_IndexValues(B), SUNSparseMatrix_NNZ(B) * sizeof(sunindextype));

  memcpy(index_pointers, SUNSparseMatrix_IndexPointers(B), (N + 1) * sizeof(sunindextype));

  // Set to updated
  SUNMatrix_ReSolve_SetUpdated(A, ReSolve::memory::HOST);

  // Sync to device if necessary
  if (memspace != ReSolve::memory::HOST)
  {
    SUNMatrix_ReSolve_SyncData(A, memspace);
  }

  // Delete the sparse matrix
  SUNMatDestroy(B);

  /* Set up the ReSolve workspace*/
  std::string refactor = "none"; // Refactorization method ID
#ifdef SUNDIALS_RESOLVE_BACKENDS_CUDA
  ReSolve::LinAlgWorkspaceCUDA workspace;
  refactor = "cusolverrf";
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  ReSolve::LinAlgWorkspaceHIP workspace;
  refactor = "rocsolverrf";
#else
  ReSolve::LinAlgWorkspaceCpu workspace;
  refactor = "klu";
#endif
  workspace.initializeHandles();

  /* ReSolve direct solver instatiation */
  ReSolve::SystemSolver solver(&workspace,
                               "klu",    // factorization
                               refactor, // refactorization
                               refactor, // triangular solve
                               "none",   // preconditioner (always 'none' here)
                               "fgmres"); // iterative refinement

  /* Set solver options */
  solver.getIterativeSolver().setCliParam("restart", "100");

  /* Create ReSolve linear solver */
  LS = SUNLinSol_ReSolve(&solver, A, memspace, sunctx);

  /* Run Tests */
  fails += Test_SUNLinSolInitialize(LS, 0);
  fails += Test_SUNLinSolSetup(LS, A, 0);
  fails += Test_SUNLinSolSolve(LS, A, x, b, 1000 * SUN_UNIT_ROUNDOFF, SUNTRUE, 0);

  fails += Test_SUNLinSolGetType(LS, SUNLINEARSOLVER_MATRIX_ITERATIVE, 0);
  fails += Test_SUNLinSolGetID(LS, SUNLINEARSOLVER_RESOLVE, 0);
  fails += Test_SUNLinSolLastFlag(LS, 0);

  // /* Use zero intial guess */
  // N_VScale(0, y, x);

  // N_VPrint_Serial(x);
  // /* Attempt to solve */
  // fails += SUNLinSolSolve_ReSolve(LS, A, x,
  //                             b, 1000 * SUN_UNIT_ROUNDOFF);

  // fails += 1;        

  /* Print result */
  if (fails)
  {
    printf("FAIL: SUNLinSol module failed %i tests \n \n", fails);
    printf("\nA =\n");
    SUNMatrix_ReSolve_Print(A, std::cout, 0);
    printf("\nx (original) =\n");
    N_VPrint_Serial(y);
    printf("\nb =\n");
    N_VPrint_Serial(b);
    printf("\nx (computed) =\n");
    N_VPrint_Serial(x);
  }
  else 
  { std::cout<< "\nSUCCESS: SUNLinSol module passed all tests on " 
                  <<  hwbackend << " backend\n\n"; 
  }

  /* Free solver, matrix and vectors */
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(x);
  N_VDestroy(y);
  N_VDestroy(b);

  SUNContext_Free(&sunctx);

  return (fails);
}

/* ----------------------------------------------------------------------
 * Implementation-specific 'check' routines
 * --------------------------------------------------------------------*/
int check_vector(N_Vector X, N_Vector Y, sunrealtype tol)
{
  int failure = 0;
  sunindextype i, local_length, maxloc;
  sunrealtype *Xdata, *Ydata, maxerr;

  Xdata        = N_VGetArrayPointer(X);
  Ydata        = N_VGetArrayPointer(Y);
  local_length = N_VGetLength_Serial(X);

  /* check vector data */
  for (i = 0; i < local_length; i++)
  {
    failure += SUNRCompareTol(Xdata[i], Ydata[i], tol);
  }

  if (failure > ZERO)
  {
    maxerr = ZERO;
    maxloc = -1;
    for (i = 0; i < local_length; i++)
    {
      if (SUNRabs(Xdata[i] - Ydata[i]) > maxerr)
      {
        maxerr = SUNRabs(Xdata[i] - Ydata[i]);
        maxloc = i;
      }
    }
    printf("check err failure: maxerr = %g at loc %li (tol = %g)\n", maxerr,
           (long int)maxloc, tol);
    return (1);
  }
  else { return (0); }
}

void sync_device() {}
