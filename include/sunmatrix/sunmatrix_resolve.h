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
 * This is the header file for the sparse implementation of the
 * SUNMATRIX module, SUNMATRIX_RESOLVE.
 * -----------------------------------------------------------------
 */

#ifndef _SUNMATRIX_RESOLVE_H
#define _SUNMATRIX_RESOLVE_H

#include <stdio.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_memory.h>
#include <resolve/matrix/Csr.hpp> 
#include <resolve/matrix/Csc.hpp>         
#include <resolve/matrix/Coo.hpp>   
#include <resolve/workspace/LinAlgWorkspace.hpp>


#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

typedef enum 
{
  SUN_RESOLVE_COO,
  SUN_RESOLVE_CSC,
  SUN_RESOLVE_CSR
} SUNMatrix_ReSolve_StorageType;

struct _SUNMatrixContent_ReSolve
{
  sunindextype M;
  sunindextype N;
  sunindextype NNZ;
  sunindextype NP;
  SUNMatrix_ReSolve_StorageType storageType;
  ReSolve::memory::MemorySpace memspace;
  ReSolve::matrix::Sparse* mat;
};

typedef struct _SUNMatrixContent_ReSolve* SUNMatrixContent_ReSolve;

/* ---------------------------------------
 * Implementation specific functions
 * ---------------------------------------*/

SUNDIALS_EXPORT SUNMatrix SUNMatrix_ReSolve(sunindextype M, sunindextype N, sunindextype NNZ,
                                            SUNMatrix_ReSolve_StorageType storageType,
                                            ReSolve::memory::MemorySpace memspace,
                                            SUNContext sunctx);

SUNDIALS_EXPORT sunindextype SUNMatrix_ReSolve_GetRows(SUNMatrix A);

SUNDIALS_EXPORT sunindextype SUNMatrix_ReSolve_GetColumns(SUNMatrix A);

SUNDIALS_EXPORT sunindextype SUNMatrix_ReSolve_GetNNZ(SUNMatrix A);

SUNDIALS_EXPORT sunindextype SUNMatrix_ReSolve_GetNP(SUNMatrix A);

SUNDIALS_EXPORT SUNMatrix_ReSolve_StorageType SUNMatrix_ReSolve_GetStorageType(SUNMatrix A);

SUNDIALS_EXPORT sunrealtype* SUNMatrix_ReSolve_GetData(SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT sunindextype* SUNMatrix_ReSolve_GetRowData(SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT sunindextype* SUNMatrix_ReSolve_GetColData(SUNMatrix A, ReSolve::memory::MemorySpace memspace);    

SUNDIALS_EXPORT SUNErrCode SUNMatrix_ReSolve_SetUpdated(SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT SUNErrCode SUNMatrix_ReSolve_SyncData(SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT void SUNMatrix_ReSolve_Print(SUNMatrix A);    

//SUNDIALS_EXPORT void SUNMatrix_ReSolve_Print_Array(SUNMatrix A);

/* ---------------------------------------
 * SUNMatrix API functions
 * ---------------------------------------*/

static inline SUNMatrix_ID SUNMatGetID_ReSolve(SUNMatrix A)
{
  return SUNMATRIX_RESOLVE;
}

SUNDIALS_EXPORT void SUNMatDestroy_ReSolve(SUNMatrix A);
SUNDIALS_EXPORT SUNErrCode SUNMatZero_ReSolve(SUNMatrix A);
SUNDIALS_EXPORT SUNMatrix SUNMatClone_ReSolve(SUNMatrix A);

#ifdef __cplusplus
}
#endif

#endif
