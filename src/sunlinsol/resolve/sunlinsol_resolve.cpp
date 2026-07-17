/* -----------------------------------------------------------------
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
 * ----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_math.h>
#include <sunlinsol/sunlinsol_resolve.hpp>
#include <sunmatrix/sunmatrix_resolve.hpp>

#include "sundials_cli.h"
#include "sundials_macros.h"

#include <resolve/vector/Vector.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/LinSolverDirect.hpp>

// GPU Vector Implementations
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
#include <nvector/nvector_cuda.h>
#include <sunmemory/sunmemory_cuda.h>
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
#include <nvector/nvector_hip.h>
#include <sunmemory/sunmemory_hip.h>
#endif

// Check for a valid precision
#if defined(SUNDIALS_EXTENDED_PRECISION)
#error \
  "Re::Solve set precision does not match SUNDIALS precision for floating type"
#endif

#if defined(SUNDIALS_INT64_T)
#error \
  "Re::Solve is using 32-bit, while SUNDIALS uses 64-bit precision for matrix indices"
#endif


#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

/*
 * -----------------------------------------------------------------
 * Re::Solve solver structure accessibility macros:
 * -----------------------------------------------------------------
 */
#define RESOLVE_CONTENT(S)  ((SUNLinearSolverContent_ReSolve)(S->content))
#define FACTORIZED(S)       (RESOLVE_CONTENT(S)->factorized)
#define LASTFLAG(S)         (RESOLVE_CONTENT(S)->last_flag)
#define RESOLVE_MEMSPACE(S) (RESOLVE_CONTENT(S)->memspace)

/*
 * ----------------------------------------------------------------------------
 * Implementation specific routines
 * ----------------------------------------------------------------------------
 */

/*
 * Constructor functions
 */

SUNLinearSolver SUNLinSol_ReSolve(ReSolve::SystemSolver* solver, SUNMatrix A, 
                                  ReSolve::memory::MemorySpace memspace, SUNContext sunctx)
{
  SUNLinearSolver S;
  SUNLinearSolverContent_ReSolve content;
  SUNMatrixContent_ReSolve A_content;

  /* Check inputs */
  if (solver == NULL || A == NULL) { return (NULL); }

  if (A->ops == NULL) { return (NULL); }

  if ( A->ops->getid == NULL)
  {
    return (NULL);
  }

  /* Check compatibility with supplied SUNMatrix */
  if (SUNMatGetID(A) != SUNMATRIX_RESOLVE) { return (NULL); }

  if (A->content == NULL) { return (NULL); }

  A_content = (SUNMatrixContent_ReSolve)A->content;
  solver->setMatrix(A_content->mat);

  /* Create the linear solver */
  S = NULL;
  S = SUNLinSolNewEmpty(sunctx);
  if (S == NULL) { return (NULL); }

  /* Attach operations */
  S->ops->gettype    = SUNLinSolGetType_ReSolve;
  S->ops->getid      = SUNLinSolGetID_ReSolve;
  S->ops->initialize = SUNLinSolInitialize_ReSolve;
  S->ops->setup      = SUNLinSolSetup_ReSolve;
  S->ops->solve      = SUNLinSolSolve_ReSolve;
  S->ops->lastflag   = SUNLinSolLastFlag_ReSolve;
  S->ops->free       = SUNLinSolFree_ReSolve;

  /* Create content */
  content = NULL;
  content = (SUNLinearSolverContent_ReSolve)malloc(sizeof(*content));
  if (content == NULL)
  {
    SUNLinSolFree(S);
    return (NULL);
  }

  /* Attach content */
  S->content = content;

  /* Fill content */
  content->last_flag = 0;
  content->solver = solver;
  content->factorized = SUNFALSE;
  content->memspace = memspace;

  return S;
}

/*
 * -----------------------------------------------------------------
 * Implementation of generic SUNLinearSolver operations.
 * -----------------------------------------------------------------
 */

SUNLinearSolver_Type SUNLinSolGetType_ReSolve(SUNLinearSolver S)
{
  auto* solver = reinterpret_cast<ReSolve::SystemSolver*>(RESOLVE_CONTENT(S)->solver);
  // The SUNLINEARSOLVER_MATRIX_ITERATIVE required methods are not fully implemented yet
  if (solver->getSolveMethod() == "randgmres" || solver->getSolveMethod() == "fgmres")
  {
   return (SUNLINEARSOLVER_MATRIX_ITERATIVE);
  }
  // Otherwise, solve method is KLU
  else
  { 
    return (SUNLINEARSOLVER_DIRECT);
  }
}

SUNLinearSolver_ID SUNLinSolGetID_ReSolve(SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_RESOLVE);
}

SUNErrCode SUNLinSolInitialize_ReSolve(SUNLinearSolver S)
{
  /* All solver-specific memory has already been allocated */
  LASTFLAG(S) = SUN_SUCCESS;
  return SUN_SUCCESS;
}


int SUNLinSolSetup_ReSolve(SUNLinearSolver S, SUNMatrix A)
{
  int status = 0;

  /* Check for valid inputs */

  if (A == NULL)
  {
    LASTFLAG(S) = SUN_ERR_ARG_CORRUPT;
    return SUN_ERR_ARG_CORRUPT;
  }

  /* Ensure that A is a ReSolve matrix */
  if (SUNMatGetID(A) != SUNMATRIX_RESOLVE)
  {
    LASTFLAG(S) = SUN_ERR_ARG_INCOMPATIBLE;
    return SUN_ERR_ARG_INCOMPATIBLE;
  }

  /* Get Re::Solve solver */
  auto* solver = reinterpret_cast<ReSolve::SystemSolver*>(RESOLVE_CONTENT(S)->solver);
  /* Check if factorization has been done*/
  if (FACTORIZED(S))
  {
    status = solver->refactorize();
    /* Check if successful */
    if (status)
    {
      LASTFLAG(S) = SUN_ERR_EXT_FAIL;
      return (LASTFLAG(S));
    }
  }
  else
  {
    status = solver->analyze();
    if (status)
    {
      LASTFLAG(S) = SUN_ERR_EXT_FAIL;
      return (LASTFLAG(S));
    }
    status = solver->factorize();
    if (status)
    {
      LASTFLAG(S) = SUN_ERR_EXT_FAIL;
      return (LASTFLAG(S));
    }
    // Perform setup only if working with GPU
    if (RESOLVE_MEMSPACE(S) != ReSolve::memory::HOST)
    {
      status = solver->refactorizationSetup();
      if (status)
      {
        LASTFLAG(S) = SUN_ERR_EXT_FAIL;
        return (LASTFLAG(S));
      }
      // Force a refactorize to work on GPU
      status = solver->refactorize();
    }
    FACTORIZED(S) = SUNTRUE;
  }

  return SUN_SUCCESS;
}

int SUNLinSolSolve_ReSolve(SUNLinearSolver S, SUNMatrix A, N_Vector x,
                              N_Vector b, sunrealtype tol)
{
  /* Check for valid inputs */
  if (S == NULL) { return SUN_ERR_ARG_CORRUPT; }

  if ((A == NULL) || (x == NULL) || (b == NULL))
  {
    LASTFLAG(S) = SUN_ERR_ARG_CORRUPT;
    return SUN_ERR_ARG_CORRUPT;
  }

  /* Ensure that A is a ReSolve matrix */
  if (SUNMatGetID(A) != SUNMATRIX_RESOLVE)
  {
    LASTFLAG(S) = SUN_ERR_ARG_INCOMPATIBLE;
    return SUN_ERR_ARG_INCOMPATIBLE;
  }

  /* Get Re::Solve solver */
  auto* solver = reinterpret_cast<ReSolve::SystemSolver*>(RESOLVE_CONTENT(S)->solver);

  /* Set tolerance if an iterative solver is set */
  if (solver->getRefinementMethod() == "fgmres" || solver->getSolveMethod() == "randgmres" || solver->getSolveMethod() == "fgmres")
  {
    solver->getIterativeSolver().setTol(tol);
  }

  /* Create vector wrappers on stack */
  sunindextype vec_length = SUNMatrix_ReSolve_Columns(A);
  ReSolve::vector::Vector vec_b(vec_length);
  ReSolve::vector::Vector vec_x(vec_length);
  
  /* Allocate data */
  vec_b.setData(N_VGetArrayPointer(b), ReSolve::memory::HOST);
  vec_x.setData(N_VGetArrayPointer(x), ReSolve::memory::HOST);
  /* Allocate to DEVICE if necessary */
  if (RESOLVE_MEMSPACE(S) != ReSolve::memory::HOST)
  {
    vec_b.setData(N_VGetDeviceArrayPointer(b), RESOLVE_MEMSPACE(S));
    vec_x.setData(N_VGetDeviceArrayPointer(x), RESOLVE_MEMSPACE(S));
  }

  /* Solve for x */
  LASTFLAG(S) = solver->solve(&vec_b, &vec_x);

  return LASTFLAG(S);
}

sunindextype SUNLinSolLastFlag_ReSolve(SUNLinearSolver S)
{
  return (LASTFLAG(S));
}

SUNErrCode SUNLinSolFree_ReSolve(SUNLinearSolver S)
{
  /* return if S is already free */
  if (S == NULL) { return SUN_SUCCESS; }

  /* delete items from contents, then delete generic structure */
  if (S->content)
  {
    free(S->content);
    S->content = NULL;
  }
  if (S->ops)
  {
    free(S->ops);
    S->ops = NULL;
  }
  free(S);
  S = NULL;
  return SUN_SUCCESS;
}
