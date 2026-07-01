/*
 * -----------------------------------------------------------------
 * Programmer(s): Jeffery Zhang
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
 * This is the testing routine to check the SUNMatrix ReSolve module
 * implementation.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_resolve.hpp>

#include "test_sunmatrix.h"

/* ----------------------------------------------------------------------
 * Main SUNMatrix Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  int fails = 0; /* counter for test failures  */
  SUNMatrix A;
  sunindextype i, j, k, kstart, kend, N, uband, lband;
  int print_timing, square;
  SUNContext sunctx;

  // Default backend
  std::string hwbackend = "CPU";

  if (SUNContext_Create(SUN_COMM_NULL, &sunctx))
  {
    printf("ERROR: SUNContext_Create failed\n");
    return (-1);
  }

  // Initialize a ReSolve HOST memory space.
  ReSolve::memory::MemorySpace memspace = ReSolve::memory::HOST;
// Check if a GPU backend is enabled
#ifdef SUNDIALS_RESOLVE_BACKENDS_CUDA
  hwbackend = "CUDA";
  memspace  = ReSolve::memory::DEVICE;
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  hwbackend = "HIP";
  memspace  = ReSolve::memory::DEVICE;
#endif

  // Create a SUNMatrix_ReSolve object
  A = NULL;
  A = SUNMatrix_ReSolve(5, 5, 13, memspace, sunctx);

  // Get pointers to content arrays
  sunrealtype* data = SUNMatrix_ReSolve_Data(A, ReSolve::memory::HOST);
  sunindextype* index_values =
    SUNMatrix_ReSolve_IndexValues(A, ReSolve::memory::HOST);
  sunindextype* index_pointers =
    SUNMatrix_ReSolve_IndexPointers(A, ReSolve::memory::HOST);

  // Fill the matrix as a 5x5 second difference matrix
  for (i = 0; i < SUNMatrix_ReSolve_NNZ(A); i++)
  {
    if (i % 3 == 0) { data[i] = 2; }
    else { data[i] = -1; }
  }

  // Row pointers
  index_pointers[0] = 0; // row 0 starts at 0  (2 non-zeros: diag + right)
  index_pointers[1] = 2; // row 1 starts at 2  (3 non-zeros: left + diag + right)
  index_pointers[2] = 5;  // row 2 starts at 5  (3 non-zeros)
  index_pointers[3] = 8;  // row 3 starts at 8  (3 non-zeros)
  index_pointers[4] = 11; // row 4 starts at 11 (2 non-zeros: left + diag)
  index_pointers[5] = 13; // total non-zeros

  // Column indices
  index_values[0]  = 0;
  index_values[1]  = 1; // row 0: cols 0, 1
  index_values[2]  = 0;
  index_values[3]  = 1;
  index_values[4]  = 2; // row 1: cols 0, 1, 2
  index_values[5]  = 1;
  index_values[6]  = 2;
  index_values[7]  = 3; // row 2: cols 1, 2, 3
  index_values[8]  = 2;
  index_values[9]  = 3;
  index_values[10] = 4; // row 3: cols 2, 3, 4
  index_values[11] = 3;
  index_values[12] = 4; // row 4: cols 3, 4

  SUNMatrix_ReSolve_SetUpdated(A, ReSolve::memory::HOST);
  // Sync host to device if necessary
  if (memspace != ReSolve::memory::HOST)
  {
    SUNMatrix_ReSolve_SyncData(A, memspace);
  }

  // Print the matrix
  printf("\n The SUNMatrix object\n");
  SUNMatrix_ReSolve_Print(A, std::cout, 0);
  printf("\n");

  std::cout << "Running SUNMatrix generic unit tests\n";
  /* SUNMatrix Tests */
  fails += Test_SUNMatGetID(A, SUNMATRIX_RESOLVE, 0);
  fails += Test_SUNMatZero(A, 0);

  if (fails)
  {
    std::cout << "FAIL: SUNMatrix module failed " << fails << " tests on the "
              << hwbackend << " hardware backend\n \n";
    printf("\nA =\n");
    SUNMatrix_ReSolve_Print(A, std::cout, 0);
  }
  else
  {
    std::cout << "\nSUCCESS: SUNMatrix module passed all tests on the "
              << hwbackend << " hardware backend\n\n";
  }

  // Destroy the SUNMatrix_ReSolve object
  SUNMatDestroy_ReSolve(A);

  return 0;
}

/* ----------------------------------------------------------------------
 * Check matrix
 * --------------------------------------------------------------------*/

// Assumes already synced
int check_matrix(SUNMatrix A, SUNMatrix B, sunrealtype tol)
{
  int failure = 0;
  sunindextype i, A_NP, B_NP, A_nnz, B_nnz;
  sunrealtype *A_data, *B_data;
  sunindextype *A_index_values, *A_index_pointers, *B_index_values,
    *B_index_pointers;

  // Get pointers to the data, pointer and value arrays
  A_data           = SUNMatrix_ReSolve_Data(A, ReSolve::memory::HOST);
  A_index_values   = SUNMatrix_ReSolve_IndexValues(A, ReSolve::memory::HOST);
  A_index_pointers = SUNMatrix_ReSolve_IndexPointers(A, ReSolve::memory::HOST);

  B_data           = SUNMatrix_ReSolve_Data(B, ReSolve::memory::HOST);
  B_index_values   = SUNMatrix_ReSolve_IndexValues(B, ReSolve::memory::HOST);
  B_index_pointers = SUNMatrix_ReSolve_IndexPointers(B, ReSolve::memory::HOST);

  // Get nnz and np
  A_nnz = SUNMatrix_ReSolve_NNZ(A);
  A_NP  = SUNMatrix_ReSolve_Rows(A);

  B_nnz = SUNMatrix_ReSolve_NNZ(B);
  B_NP  = SUNMatrix_ReSolve_Rows(B);

  // Check same storage type
  if (SUNMatGetID(A) != SUNMatGetID(B))
  {
    printf(">>> ERROR: check_matrix: Different storage types (%d vs %d)\n",
           SUNMatGetID(A), SUNMatGetID(B));
    return (1);
  }

  // Check shape
  if (SUNMatrix_ReSolve_Rows(A) != SUNMatrix_ReSolve_Rows(B))
  {
    printf(">>> ERROR: check_matrix: Different numbers of rows (%ld vs %ld)\n",
           (long int)SUNMatrix_ReSolve_Rows(A),
           (long int)SUNMatrix_ReSolve_Rows(B));
    return (1);
  }
  if (SUNMatrix_ReSolve_Columns(A) != SUNMatrix_ReSolve_Columns(B))
  {
    printf(">>> ERROR: check_matrix: Different numbers of columns (%ld vs "
           "%ld)\n",
           (long int)SUNMatrix_ReSolve_Columns(A),
           (long int)SUNMatrix_ReSolve_Columns(B));
    return (1);
  }

  // Check non zeros
  if (A_nnz != B_nnz)
  {
    printf(">>> ERROR: check_matrix: Different numbers of nonzeros (%ld vs "
           "%ld)\n",
           (long int)A_nnz, (long int)B_nnz);
    return (1);
  }

  /* compare sparsity patterns */
  for (i = 0; i < A_NP; i++)
  {
    failure += (A_index_pointers[i] != B_index_pointers[i]);
  }

  if (failure > ZERO)
  {
    printf(">>> ERROR: check_matrix: Different indexptrs \n");
    return (1);
  }

  for (i = 0; i < A_nnz; i++)
  {
    failure += (A_index_values[i] != B_index_values[i]);
  }

  if (failure > ZERO)
  {
    printf(">>> ERROR: check_matrix: Different indexvals \n");
    return (1);
  }

  /* compare matrix values */
  for (i = 0; i < A_nnz; i++)
  {
    failure += SUNRCompareTol(A_data[i], B_data[i], tol);
  }
  if (failure > ZERO)
  {
    printf(">>> ERROR: check_matrix: Different entries \n");
    return (1);
  }

  return (0);
}

int check_matrix_entry(SUNMatrix A, sunrealtype val, sunrealtype tol)
{
  int failure = 0;
  sunrealtype* Adata;
  sunindextype* indexptrs;
  sunindextype i, NP;

  /* get data pointer */
  Adata = SUNMatrix_ReSolve_Data(A, ReSolve::memory::HOST);

  /* compare data */
  indexptrs = SUNMatrix_ReSolve_IndexPointers(A, ReSolve::memory::HOST);
  NP        = SUNMatrix_ReSolve_Rows(A);
  for (i = 0; i < indexptrs[NP]; i++)
  {
    failure += SUNRCompareTol(Adata[i], val, tol);
  }

  if (failure > ZERO) { return (1); }
  else { return (0); }
}

int check_vector(N_Vector actual, N_Vector expected, sunrealtype tol)
{
  return 0;
}

sunbooleantype has_data(SUNMatrix A)
{
  sunrealtype* Adata = SUNMatrix_ReSolve_Data(A, ReSolve::memory::HOST);
  if (Adata == NULL) { return SUNFALSE; }
  else { return SUNTRUE; }
}

sunbooleantype is_square(SUNMatrix A)
{
  if (SUNMatrix_ReSolve_Rows(A) == SUNMatrix_ReSolve_Columns(A))
  {
    return SUNTRUE;
  }
  else { return SUNFALSE; }
}

void sync_device(SUNMatrix A)
{
  /* sync is performed by functions for now */
  return;
}