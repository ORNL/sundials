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
#include <sunmatrix/sunmatrix_resolve.h>

#include "test_sunmatrix.h"

/* ----------------------------------------------------------------------
 * Main SUNMatrix Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  int fails = 0;                   /* counter for test failures  */
//   sunindextype matrows, matcols;   /* matrix dims                */
  SUNMatrix A;
  sunindextype i, j, k, kstart, kend, N, uband, lband;
  int print_timing, square;
  SUNContext sunctx;

  if (SUNContext_Create(SUN_COMM_NULL, &sunctx))
  {
    printf("ERROR: SUNContext_Create failed\n");
    return (-1);
  }

//   /* check input and set vector length */
//   if (argc < 5)
//   {
//     printf("ERROR: FOUR (4) Input required: matrix rows, matrix cols, matrix "
//            "type (0/1), print timing \n");
//     return (-1);
//   }

//   matrows = (sunindextype)atol(argv[1]);
//   if (matrows < 1)
//   {
//     printf("ERROR: number of rows must be a positive integer\n");
//     return (-1);
//   }

//   matcols = (sunindextype)atol(argv[2]);
//   if (matcols < 1)
//   {
//     printf("ERROR: number of cols must be a positive integer\n");
//     return (-1);
//   }

  // Initialize a ReSolve HOST memory space.
  ReSolve::memory::MemorySpace memspace = ReSolve::memory::HOST;
  // Create a SUNMatrix_ReSolve object
  A = NULL;
  A = SUNMatrix_ReSolve(5, 5, 13, memspace, sunctx);

  // Get pointers to content arrays
  sunrealtype* data = SUNMatrix_ReSolve_Data(A, memspace);
  sunindextype* index_values = SUNMatrix_ReSolve_IndexValues(A, memspace);
  sunindextype* index_pointers = SUNMatrix_ReSolve_IndexPointers(A, memspace);

  // Fill the matrix as a 5x5 second difference matrix
  for (i = 0; i < SUNMatrix_ReSolve_NNZ(A); i++)
  {
    if (i % 3 == 0)
    {
      data[i] = 2;
    }
    else
    {
      data[i] = -1;
    }
  }

  // Row pointers — how many non-zeros before each row
  index_pointers[0] = 0;   // row 0 starts at 0  (2 non-zeros: diag + right)
  index_pointers[1] = 2;   // row 1 starts at 2  (3 non-zeros: left + diag + right)
  index_pointers[2] = 5;   // row 2 starts at 5  (3 non-zeros)
  index_pointers[3] = 8;   // row 3 starts at 8  (3 non-zeros)
  index_pointers[4] = 11;  // row 4 starts at 11 (2 non-zeros: left + diag)
  index_pointers[5] = 13;  // total non-zeros

  // Column indices
  index_values[0]  = 0; index_values[1]  = 1;              // row 0: cols 0, 1
  index_values[2]  = 0; index_values[3]  = 1; index_values[4]  = 2; // row 1: cols 0, 1, 2
  index_values[5]  = 1; index_values[6]  = 2; index_values[7]  = 3; // row 2: cols 1, 2, 3
  index_values[8]  = 2; index_values[9]  = 3; index_values[10] = 4; // row 3: cols 2, 3, 4
  index_values[11] = 3; index_values[12] = 4;              // row 4: cols 3, 4

  // Print the second difference matrix
  SUNMatrix_ReSolve_Print(A, memspace);

  // Set the matrix to zero
  SUNMatZero_ReSolve(A);

  // Print empty matrix
  SUNMatrix_ReSolve_Print(A, memspace);

  // Destroy the SUNMatrix_ReSolve object
  SUNMatDestroy_ReSolve(A);

  return 0;
}

/* ----------------------------------------------------------------------
 * Check matrix
 * --------------------------------------------------------------------*/

// Currently stubs
int check_matrix(SUNMatrix A, SUNMatrix B, sunrealtype tol)
{
  int failure = 0;
  sunrealtype *Adata, *Bdata;
  sunindextype *Aindexptrs, *Bindexptrs;
  sunindextype *Aindexvals, *Bindexvals;
  sunindextype i, ANP, BNP, Annz, Bnnz;

  return (0);
}

int check_matrix_entry(SUNMatrix A, sunrealtype val, sunrealtype tol)
{
  return (0);
}

int check_vector(N_Vector x, N_Vector y, sunrealtype tol)
{
  return 0;
}

sunbooleantype has_data(SUNMatrix A)
{
  return SUNTRUE;
}

sunbooleantype is_square(SUNMatrix A)
{
  if (SUNMatrix_ReSolve_Rows(A) == SUNMatrix_ReSolve_Columns(A)) { return SUNTRUE; }
  else { return SUNFALSE; }
}

void sync_device(SUNMatrix A)
{
  /* not running on GPU, just return */
  return;
}