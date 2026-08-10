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
 * This is the header file for the Re::Solve implementation of the
 * SUNLINSOL module, SUNLINSOL_RESOLVE.
 * -----------------------------------------------------------------
 */

#ifndef _SUNLINSOL_RESOLVE_HPP
#define _SUNLINSOL_RESOLVE_HPP

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_memory.h>
#include <sundials/sundials_nvector.h>

#include <resolve/MemoryUtils.hpp>

/* -----------------------------------------------
 * ReSolve implementation of SUNLinearSolver
 * ----------------------------------------------- */

/* Forward Declaration of the SystemSolver class */
namespace ReSolve {
class SystemSolver;
}

struct _SUNLinearSolverContent_ReSolve
{
  int last_flag;
  ReSolve::SystemSolver* solver;
  sunbooleantype factorized;
  ReSolve::memory::MemorySpace memspace;
};

typedef struct _SUNLinearSolverContent_ReSolve* SUNLinearSolverContent_ReSolve;

SUNDIALS_EXPORT SUNLinearSolver
SUNLinSol_ReSolve(ReSolve::SystemSolver* solver, SUNMatrix A,
                  ReSolve::memory::MemorySpace memspace, SUNContext sunctx);

SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_ReSolve(SUNLinearSolver S);
SUNDIALS_EXPORT SUNLinearSolver_ID SUNLinSolGetID_ReSolve(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolInitialize_ReSolve(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSetup_ReSolve(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_ReSolve(SUNLinearSolver S, SUNMatrix A,
                                           N_Vector x, N_Vector b,
                                           sunrealtype tol);
SUNDIALS_EXPORT sunindextype SUNLinSolLastFlag_ReSolve(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolFree_ReSolve(SUNLinearSolver S);

#endif