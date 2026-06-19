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
#define RESOLVE_CONTENT(A)    ((SUNMatrixContent_ReSolve)(A)->content) // ASSUMES CSR FORMAT DEFAULT
#define RESOLVE_MAT(A)        (RESOLVE_CONTENT(A)->mat)
#define RESOLVE_M(A)          (RESOLVE_CONTENT(A)->M)
#define RESOLVE_N(A)          (RESOLVE_CONTENT(A)->N)
#define RESOLVE_NP(A)         (RESOLVE_CONTENT(A)->NP)
#define RESOLVE_NNZ(A)        (RESOLVE_CONTENT(A)->NNZ)
#define RESOLVE_MEMSPACE(A)   (RESOLVE_CONTENT(A)->memspace)

/* --------------------------------------------------------------------------
 * Constructor
 * -------------------------------------------------------------------------- */

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
  A->ops->clone     = SUNMatClone_ReSolve;

  // Create ReSolve matrix
  ReSolve::matrix::Csr* mat = new ReSolve::matrix::Csr(m, n, nnz);
  mat->allocateMatrixData(ReSolve::memory::HOST);
  // Allocate matrix on device if necessary
  if (memspace == ReSolve::memory::DEVICE)
  {
    mat->allocateMatrixData(memspace);
  }

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
  content->memspace   = memspace;


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

 @param[in] A The SUNMatrix object
*/
sunrealtype* SUNMatrix_ReSolve_Data(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  return RESOLVE_MAT(A)->getValues(memspace);
}

/**
 Get the pointer to the ReSolve matrix offsets array

 @param[in] A The SUNMatrix object
*/
sunindextype* SUNMatrix_ReSolve_IndexPointers(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  return RESOLVE_MAT(A)->getRowData(memspace);
}

/**
 Get the pointer to the ReSolve matrix indices array

 @param[in] A The SUNMatrix object
*/
sunindextype* SUNMatrix_ReSolve_IndexValues(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  return RESOLVE_MAT(A)->getColData(memspace);
}

/**
  * @brief Tags `memspace` as updated.
  *
  * @param[in] memspace - memory space (HOST or DEVICE) to set to "updated"
  *
  * @return 0 if successful, -1 if not.
  *
  * The method sets the boolean flag indicating that the `memspace` is updated.
  * It automatically sets the other data mirror to non-updated. You would
  * use this function if you update matrix data by accessing its raw pointers.
  * In such case, the matrix has no way of knowing which data is most recent, so
  * you have to tell it.
  *
  * @warning This is an expert-level function. Use only if you know what you are
  * doing.
  *
  * @note If you want to set both DEVICE and HOST memory to the same value
  * use syncData function.
*/
SUNErrCode SUNMatrix_ReSolve_SetUpdated(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  RESOLVE_MAT(A)->setUpdated(memspace);
  return SUN_SUCCESS;
}

/**
  * @brief Sync data in memspace with the updated memory space.
  *
  * @param A - The SUNMatrix_ReSolve object
  * @param memspace - memory space to be synced up (HOST or DEVICE)
  *
  * @pre The memory space other than `memspace` must be up-to-date. Otherwise,
  * this function will return an error.
  *
  * @see Sparse::setUpdated in the Re::Solve library
*/
SUNErrCode SUNMatrix_ReSolve_SyncData(SUNMatrix A, ReSolve::memory::MemorySpace memspace)
{
  RESOLVE_MAT(A)->syncData(memspace);
  return SUN_SUCCESS;
}

/** 
 Print the Re::Solve matrix
*/
void SUNMatrix_ReSolve_Print(SUNMatrix A)
{
  RESOLVE_MAT(A)->print(std::cout, 0);
}

/**
  Utility function to print an array on the device for debugging
*/
// void SUNMatrix_ReSolve_Print_Array(SUNMatrix A)
// {
//   if (RESOLVE_MEMSPACE(A) == ReSolve::memory::HOST) {return;}

//   ReSolve::memory::MemorySpace memspace = RESOLVE_MEMSPACE(A);
//   sunrealtype* d_data = SUNMatrix_ReSolve_Data(A, memspace);
//   sunrealtype* h_data = new sunrealtype[SUNMatrix_ReSolve_NNZ(A)];

//   cudaMemcpy(h_data, d_data, 
//            SUNMatrix_ReSolve_NNZ(A) * sizeof(sunrealtype),
//            cudaMemcpyDeviceToHost);

//   for (int i = 0; i < SUNMatrix_ReSolve_NNZ(A); i++)
//   {
//     printf("data[%d] = %f\n", i, h_data[i]);
//   }
//   delete[] h_data;
// }

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

// This function automatically syncs data between host and device
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

  // Get pointers to the data, indexvalues and indexpointers arrays on host in ReSolve
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
    index_pointers[i] = ZERO; 
  }
  
  (index_pointers)[RESOLVE_NP(A)] = 0;

  SUNMatrix_ReSolve_SetUpdated(A, ReSolve::memory::HOST);

  // Sync to device if necessary
  if (RESOLVE_MEMSPACE(A) != ReSolve::memory::HOST) 
  { 
    SUNMatrix_ReSolve_SyncData(A, RESOLVE_MEMSPACE(A));
  }

  return SUN_SUCCESS;
}

SUNMatrix SUNMatClone_ReSolve(SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);
  SUNMatrix B = SUNMatrix_ReSolve(RESOLVE_M(A), RESOLVE_N(A), RESOLVE_NNZ(A),
                                RESOLVE_MEMSPACE(A), A->sunctx);
  SUNCheckLastErrNull();
  return (B);
}
