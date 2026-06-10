/* ---------------------------------------------------------------------------
 * Programmer(s): Jeffery Zhang
 * ---------------------------------------------------------------------------
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
 * ---------------------------------------------------------------------------
 * This is the implementation file for the dense implementation of the
 * SUNMATRIX class using the Re::Solve library
 * ---------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>

// Re::Solve headers
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/MemoryUtils.hpp>

// SUNDIALS headers
#include <sunmatrix/sunmatrix_resolve.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_math.h>
#include "sundials/sundials_errors.h"
#include "sundials_debug.h"
#include "sundials_macros.h"

// Check for a valid precision
#if defined(SUNDIALS_EXTENDED_PRECISION)
#error "Re::Solve set precision does not match SUNDIALS precision for floating type"
#endif

#if defined(SUNDIALS_INT64_T)
#error "Re::Solve set precision does not match SUNDIALS precision for indices"
#endif

// Constants
#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

// Content accessor macro
#define RESOLVE_CONTENT(A) ((SUNMatrixContent_ReSolve)(A)->content) // ASSUMES CSR FORMAT DEFAULT
#define RESOLVE_MAT(A) (RESOLVE_CONTENT(A)->mat)
#define RESOLVE_M(A) (RESOLVE_CONTENT(A)->M)
#define RESOLVE_N(A) (RESOLVE_CONTENT(A)->N)
#define RESOLVE_NP(A) (RESOLVE_CONTENT(A)->NP)
#define RESOLVE_NNZ(A) (RESOLVE_CONTENT(A)->NNZ)

// TODO add checks for gpu memspace


/* --------------------------------------------------------------------------
 * Constructor
 * -------------------------------------------------------------------------- */

// TODO Add ability to choose format. Currently, constructor assumes CSR format 
SUNMatrix SUNMatrix_ReSolve(sunindextype m, sunindextype n, sunindextype nnz,
                            ReSolve::memory::MemorySpace memspace, SUNContext sunctx)
{
  SUNFunctionBegin(sunctx);
  SUNMatrixContent_ReSolve content;

  // Check inputs
  if ((m <= 0) || (n <= 0) || (nnz < 0))
  {
    SUNDIALS_DEBUG_ERROR("Illegal input\n");
    return NULL;
  }

  // Create an empty matrix object
  SUNMatrix A = SUNMatNewEmpty(sunctx);
  if (!A)
  {
    SUNDIALS_DEBUG_ERROR("SUNMatNewEmpty returned NULL\n");
    return NULL;
  }

  // Attach operations
  A->ops->getid     = SUNMatGetID_ReSolve;
  A->ops->destroy   = SUNMatDestroy_ReSolve;
  A->ops->zero      = SUNMatZero_ReSolve;

  // Create ReSolve matrix
  ReSolve::matrix::Csr* mat = new ReSolve::matrix::Csr(m, n, nnz);
  mat->allocateMatrixData(memspace);

  /* Create content */
  content = NULL;
  content = (SUNMatrixContent_ReSolve)malloc(sizeof *content);
  SUNAssertNull(content, SUN_ERR_MALLOC_FAIL);

  /* Attach content */
  A->content = content;

  /* Fill content */
  content->M          = m;
  content->N          = n;
  content->NNZ        = nnz;
  content->NP         = m;
  content->mat        = mat;


  if (!(A->content))
  {
    SUNDIALS_DEBUG_ERROR("Content allocation failed\n");
    SUNMatDestroy(A);
    return NULL;
  }

  return A;
}

/* --------------------------------------------------------------------------
 * Implementation specific functions
 * -------------------------------------------------------------------------- */

sunindextype SUNMatrix_ReSolve_Rows(SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);
  SUNAssertNoRet(SUNMatGetID(A) == SUNMATRIX_RESOLVE, SUN_ERR_ARG_WRONGTYPE);
  return RESOLVE_M(A);
}

sunindextype SUNMatrix_ReSolve_Columns(SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);
  SUNAssertNoRet(SUNMatGetID(A) == SUNMATRIX_RESOLVE, SUN_ERR_ARG_WRONGTYPE);
  return RESOLVE_N(A);
}

sunindextype SUNMatrix_ReSolve_NNZ(SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);
  SUNAssertNoRet(SUNMatGetID(A) == SUNMATRIX_RESOLVE, SUN_ERR_ARG_WRONGTYPE);
  return RESOLVE_NNZ(A);
}

sunindextype SUNMatrix_ReSolve_NP(SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);
  SUNAssertNoRet(SUNMatGetID(A) == SUNMATRIX_RESOLVE, SUN_ERR_ARG_WRONGTYPE);
  return RESOLVE_NP(A);
}

/**
 Get the pointer to the ReSolve matrix data array
 The pointer is always to the array stored on the host

 @param A The SUNMatrix object
*/
sunrealtype* SUNMatrix_ReSolve_Data(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  return RESOLVE_MAT(A)->getValues(ReSolve::memory::HOST);
}

/**
 Get the pointer to the ReSolve matrix offsets array
 The pointer is always to the array stored on the host

 @param A The SUNMatrix object
*/
sunindextype* SUNMatrix_ReSolve_IndexPointers(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  return RESOLVE_MAT(A)->getRowData(ReSolve::memory::HOST);
}

/**
 Get the pointer to the ReSolve matrix indices array
 The pointer is always to the array stored on the host

 @param A The SUNMatrix object
*/
sunindextype* SUNMatrix_ReSolve_IndexValues(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  return RESOLVE_MAT(A)->getColData(ReSolve::memory::HOST);
}

/** 
 Print the Re::Solve matrix
*/
void SUNMatrix_ReSolve_Print(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  RESOLVE_MAT(A)->print(std::cout, 0);
}

/* --------------------------------------------------------------------------
 * Implementation of generic SUNMatrix operations.
 * -------------------------------------------------------------------------- */

void SUNMatDestroy_ReSolve(SUNMatrix A)
{
  if (!A)
  {
    SUNDIALS_DEBUG_ERROR("Input matrix is NULL\n");
    return;
  }

  if (SUNMatGetID(A) != SUNMATRIX_RESOLVE)
  {
    SUNDIALS_DEBUG_ERROR("Invalid matrix ID\n");
    return;
  }

  // Free ReSolve matrix
  delete RESOLVE_MAT(A);

  /* free content struct */
  free(A->content);
  A->content = NULL;

  /* free ops and matrix */
  if (A->ops)
  {
    free(A->ops);
    A->ops = NULL;
  }

  // Free matrix
  SUNMatFreeEmpty(A);
  A = NULL;

  return;
}

SUNErrCode SUNMatZero_ReSolve(SUNMatrix A)
{
  if (!A)
  {
    SUNDIALS_DEBUG_ERROR("Input matrix is NULL\n");
    return SUN_ERR_ARG_INCOMPATIBLE;
  }

  if (SUNMatGetID(A) != SUNMATRIX_RESOLVE)
  {
    SUNDIALS_DEBUG_ERROR("Invalid matrix ID\n");
    return SUN_ERR_ARG_INCOMPATIBLE;
  }

  sunindextype i;
  // TODO sync between device and data

  // Get pointers to the data, indexvalues and indexpointers arrays in ReSolve
  sunrealtype* values = RESOLVE_MAT(A)->getValues(ReSolve::memory::HOST);

  sunindextype* index_pointers = RESOLVE_MAT(A)->getRowData(ReSolve::memory::HOST);

  sunindextype* index_values = RESOLVE_MAT(A)->getColData(ReSolve::memory::HOST);
  
  // Zero out the values of these arrays
  for (i = 0; i < RESOLVE_NNZ(A); i++)
  {
    values[i]              = ZERO;
    index_values[i]        = 0;
  }

  for (i = 0; i < RESOLVE_NP(A); i++) 
  { 
    index_pointers[i] = 0; 
  }
  
  (index_pointers)[RESOLVE_NP(A)] = 0;

  return SUN_SUCCESS;
}
