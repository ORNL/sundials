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

#include <resolve/matrix/Csr.hpp>
#include <stdio.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_memory.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

struct _SUNMatrixContent_ReSolve
{
  sunindextype M;
  sunindextype N;
  sunindextype NNZ;
  ReSolve::memory::MemorySpace memspace;
  ReSolve::matrix::Csr* mat;
};

typedef struct _SUNMatrixContent_ReSolve* SUNMatrixContent_ReSolve;

/* ---------------------------------------
 * Implementation specific functions
 * ---------------------------------------*/

SUNDIALS_EXPORT SUNMatrix SUNMatrix_ReSolve(sunindextype M, sunindextype N,
                                            sunindextype NNZ,
                                            ReSolve::memory::MemorySpace memspace,
                                            SUNContext sunctx);

SUNDIALS_EXPORT sunindextype SUNMatrix_ReSolve_Rows(SUNMatrix A);

SUNDIALS_EXPORT sunindextype SUNMatrix_ReSolve_Columns(SUNMatrix A);

SUNDIALS_EXPORT sunindextype SUNMatrix_ReSolve_NNZ(SUNMatrix A);

SUNDIALS_EXPORT sunrealtype* SUNMatrix_ReSolve_Data(
  SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT sunindextype* SUNMatrix_ReSolve_IndexValues(
  SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT sunindextype* SUNMatrix_ReSolve_IndexPointers(
  SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT SUNErrCode
SUNMatrix_ReSolve_SetUpdated(SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT SUNErrCode
SUNMatrix_ReSolve_SyncData(SUNMatrix A, ReSolve::memory::MemorySpace memspace);

SUNDIALS_EXPORT void SUNMatrix_ReSolve_Print(SUNMatrix A, std::ostream& out,
                                             sunindextype indexing_base);

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
