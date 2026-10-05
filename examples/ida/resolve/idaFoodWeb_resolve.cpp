/* -----------------------------------------------------------------
 * Programmer(s): Jeffery Zhang
 *
 * Based on code by: Allan Taylor, Alan Hindmarsh and Radu Serban @ LLNL
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
 * Example program for IDA: Food web problem.
 *
 * This example program uses the ReSolve linear
 * solver, and IDACalcIC for initial condition calculation.
 *
 * The mathematical problem solved in this example is a DAE system
 * that arises from a system of partial differential equations after
 * spatial discretization. The PDE system is a food web population
 * model, with predator-prey interaction and diffusion on the unit
 * square in two dimensions. The dependent variable vector is:
 *
 *         1   2         ns
 *   c = (c , c ,  ..., c  ) , ns = 2 * np
 *
 * and the PDE's are as follows:
 *
 *     i             i      i
 *   dc /dt = d(i)*(c    + c  )  +  R (x,y,c)   (i = 1,...,np)
 *                   xx     yy       i
 *
 *              i      i
 *   0 = d(i)*(c    + c  )  +  R (x,y,c)   (i = np+1,...,ns)
 *              xx     yy       i
 *
 *   where the reaction terms R are:
 *
 *                   i             ns         j
 *   R  (x,y,c)  =  c  * (b(i)  + sum a(i,j)*c )
 *    i                           j=1
 *
 * The number of species is ns = 2 * np, with the first np being
 * prey and the last np being predators. The coefficients a(i,j),
 * b(i), d(i) are:
 *
 *  a(i,i) = -AA   (all i)
 *  a(i,j) = -GG   (i <= np , j >  np)
 *  a(i,j) =  EE   (i >  np, j <= np)
 *  all other a(i,j) = 0
 *  b(i) = BB*(1+ alpha * x*y + beta*sin(4 pi x)*sin(4 pi y)) (i <= np)
 *  b(i) =-BB*(1+ alpha * x*y + beta*sin(4 pi x)*sin(4 pi y)) (i  > np)
 *  d(i) = DPREY   (i <= np)
 *  d(i) = DPRED   (i > np)
 *
 * The various scalar parameters required are set using '#define'
 * statements or directly in routine InitUserData. In this program,
 * np = 1, ns = 2. The boundary conditions are homogeneous Neumann:
 * normal derivative = 0.
 *
 * A polynomial in x and y is used to set the initial values of the
 * first np variables (the prey variables) at each x,y location,
 * while initial values for the remaining (predator) variables are
 * set to a flat value, which is corrected by IDACalcIC.
 *
 * The PDEs are discretized by central differencing on a MX by MY
 * mesh.
 *
 * The DAE system is solved by IDA using the banded linear solver.
 * Output is printed at t = 0, .001, .01, .1, .4, .7, 1.
 * -----------------------------------------------------------------
 * References:
 * [1] Peter N. Brown and Alan C. Hindmarsh,
 *     Reduced Storage Matrix Methods in Stiff ODE systems, Journal
 *     of Applied Mathematics and Computation, Vol. 31 (May 1989),
 *     pp. 40-91.
 *
 * [2] Peter N. Brown, Alan C. Hindmarsh, and Linda R. Petzold,
 *     Using Krylov Methods in the Solution of Large-Scale
 *     Differential-Algebraic Systems, SIAM J. Sci. Comput., 15
 *     (1994), pp. 1467-1488.
 *
 * [3] Peter N. Brown, Alan C. Hindmarsh, and Linda R. Petzold,
 *     Consistent Initial Condition Calculation for Differential-
 *     Algebraic Systems, SIAM J. Sci. Comput., 19 (1998),
 *     pp. 1495-1512.
 * -----------------------------------------------------------------*/

#include <ida/ida.h>
#include <math.h>
#include <nvector/nvector_serial.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunmatrix/sunmatrix_resolve.hpp>
#include <sunlinsol/sunlinsol_resolve.hpp>

#include <resolve/SystemSolver.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
#include <nvector/nvector_cuda.h>
#include <sunmemory/sunmemory_cuda.h>
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
#include <nvector/nvector_hip.h>
#include <sunmemory/sunmemory_hip.h>
#endif

/* Problem Constants */

#define NPREY       1
#define NUM_SPECIES 2 * NPREY

#define PI     SUN_RCONST(3.1415926535898)
#define FOURPI (SUN_RCONST(4.0) * PI)

#define MX    20
#define MY    20
#define NSMX  (NUM_SPECIES * MX)
#define NEQ   (NUM_SPECIES * MX * MY)
#define AA    SUN_RCONST(1.0)
#define EE    SUN_RCONST(10000.)
#define GG    SUN_RCONST(0.5e-6)
#define BB    SUN_RCONST(1.0)
#define DPREY SUN_RCONST(1.0)
#define DPRED SUN_RCONST(0.05)
#define ALPHA SUN_RCONST(50.)
#define BETA  SUN_RCONST(1000.)
#define AX    SUN_RCONST(1.0)
#define AY    SUN_RCONST(1.0)
#define RTOL  SUN_RCONST(1.e-5)
#define ATOL  SUN_RCONST(1.e-5)
#define NOUT  6
#define TMULT SUN_RCONST(10.0)
#define TADD  SUN_RCONST(0.3)
#define ZERO  SUN_RCONST(0.)
#define ONE   SUN_RCONST(1.0)

/* IJ_Vptr works on host pointers only */
#define IJ_Vptr_h(ptr, i, j) ((ptr) + (i) * NUM_SPECIES + (j) * NSMX)

typedef struct
{
  sunindextype Neq, ns, np, mx, my;
  sunrealtype dx, dy, **acoef;
  sunrealtype cox[NUM_SPECIES], coy[NUM_SPECIES], bcoef[NUM_SPECIES];
  N_Vector rates;
}* UserData;

/* Function prototypes */
static int resweb(sunrealtype time, N_Vector cc, N_Vector cp, N_Vector resval,
                  void* user_data);
static int jacFoodInit(SUNMatrix JJ, UserData webdata);
static int jacFoodWeb(sunrealtype tt, sunrealtype cj, N_Vector cc, N_Vector cp,
                      N_Vector resvec, SUNMatrix JJ, void* user_data,
                      N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);
static void InitUserData(UserData webdata);
static void SetInitialProfiles(N_Vector cc, N_Vector cp, N_Vector id,
                               UserData webdata);
static void PrintHeader(sunrealtype rtol, sunrealtype atol,
                        std::string hwbackend);
static void PrintOutput(void* mem, N_Vector c, sunrealtype t);
static void PrintFinalStats(void* mem);
static void Fweb(sunrealtype tcalc, sunrealtype* ccv, sunrealtype* cratev,
                 UserData webdata);
static void WebRates(sunrealtype xx, sunrealtype yy, sunrealtype* cxy,
                     sunrealtype* ratesxy, UserData webdata);
static sunrealtype dotprod(sunindextype size, sunrealtype* x1, sunrealtype* x2);
static int check_retval(void* returnvalue, const char* funcname, int opt);

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

int main(void)
{
  void* mem;
  UserData webdata;
  N_Vector cc, cp, id;
  int iout, retval;
  sunrealtype rtol, atol, t0, tout, tret;
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNContext ctx;

  mem     = NULL;
  webdata = NULL;
  cc = cp = id = NULL;
  A            = NULL;
  LS           = NULL;

  retval = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_retval(&retval, "SUNContext_Create", 1)) { return 1; }

  /* Allocate vectors using GPU type if available */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  cc = N_VNew_Cuda(NEQ, ctx);
  if (check_retval((void*)cc, "N_VNew_Cuda", 0)) { return 1; }
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  cc = N_VNew_Hip(NEQ, ctx);
  if (check_retval((void*)cc, "N_VNew_Hip", 0)) { return 1; }
#else
  cc = N_VNew_Serial(NEQ, ctx);
  if (check_retval((void*)cc, "N_VNew_Serial", 0)) { return 1; }
#endif
  cp = N_VClone(cc);
  if (check_retval((void*)cp, "N_VClone", 0)) { return 1; }
  id = N_VClone(cc);
  if (check_retval((void*)id, "N_VClone", 0)) { return 1; }

  /* Allocate and initialize user data */
  webdata        = (UserData)malloc(sizeof *webdata);
  webdata->rates = N_VClone(cc);
  webdata->acoef = SUNDlsMat_newDenseMat(NUM_SPECIES, NUM_SPECIES);
  InitUserData(webdata);

  SetInitialProfiles(cc, cp, id, webdata);

  t0   = ZERO;
  rtol = RTOL;
  atol = ATOL;

  mem = IDACreate(ctx);
  if (check_retval((void*)mem, "IDACreate", 0)) { return 1; }

  retval = IDASetUserData(mem, webdata);
  if (check_retval(&retval, "IDASetUserData", 1)) { return 1; }

  retval = IDASetId(mem, id);
  if (check_retval(&retval, "IDASetId", 1)) { return 1; }

  retval = IDAInit(mem, resweb, t0, cc, cp);
  if (check_retval(&retval, "IDAInit", 1)) { return 1; }

  retval = IDASStolerances(mem, rtol, atol);
  if (check_retval(&retval, "IDASStolerances", 1)) { return 1; }

  /* Set up ReSolve memory space and backend */
  ReSolve::memory::MemorySpace memspace = ReSolve::memory::HOST;
  std::string hwbackend = "CPU";
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  hwbackend = "CUDA";
  memspace  = ReSolve::memory::DEVICE;
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  hwbackend = "HIP";
  memspace  = ReSolve::memory::DEVICE;
#endif

  /* Exact NNZ count:
       ns*ns per node (self block, full reaction coupling)
     + 2*(mx-1)*my*ns  (left/right diffusion, same-species only)
     + 2*mx*(my-1)*ns  (up/down diffusion, same-species only)    */
  sunindextype nnz = (sunindextype)MX * MY * NUM_SPECIES * NUM_SPECIES
                   + 2 * (MX - 1) * MY * NUM_SPECIES
                   + 2 * MX * (MY - 1) * NUM_SPECIES;

  A = SUNMatrix_ReSolve(NEQ, NEQ, nnz, memspace, ctx);
  if (check_retval((void*)A, "SUNMatrix_ReSolve", 0)) { return 1; }

  /* Initialize the sparsity structure */
  jacFoodInit(A, webdata);

  std::string refactor = "none";
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
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

  ReSolve::SystemSolver solver(&workspace,
                               "klu",
                               refactor,
                               refactor,
                               "none",
                               "none");
                              
  LS = SUNLinSol_ReSolve(&solver, A, memspace, ctx);
  if (check_retval((void*)LS, "SUNLinSol_ReSolve", 0)) { return 1; }

  retval = IDASetLinearSolver(mem, LS, A);
  if (check_retval(&retval, "IDASetLinearSolver", 1)) { return 1; }

  retval = IDASetJacFn(mem, jacFoodWeb);
  if (check_retval(&retval, "IDASetJacFn", 1)) { return 1; }

  tout   = SUN_RCONST(0.001);
  retval = IDACalcIC(mem, IDA_YA_YDP_INIT, tout);
  if (check_retval(&retval, "IDACalcIC", 1)) { return 1; }

  PrintHeader(rtol, atol, hwbackend);
  PrintOutput(mem, cc, ZERO);

  for (iout = 1; iout <= NOUT; iout++)
  {
    retval = IDASolve(mem, tout, &tret, cc, cp, IDA_NORMAL);
    if (check_retval(&retval, "IDASolve", 1)) { return retval; }

    PrintOutput(mem, cc, tret);

    if (iout < 3) { tout *= TMULT; }
    else { tout += TADD; }
  }

  PrintFinalStats(mem);

  IDAFree(&mem);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(cc);
  N_VDestroy(cp);
  N_VDestroy(id);
  SUNDlsMat_destroyMat(webdata->acoef);
  N_VDestroy(webdata->rates);
  free(webdata);
  SUNContext_Free(&ctx);

  return 0;
}

/* Macros for UserData fields */
#define acoef (webdata->acoef)
#define bcoef (webdata->bcoef)
#define cox   (webdata->cox)
#define coy   (webdata->coy)

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDA
 *--------------------------------------------------------------------
 */

static int resweb(sunrealtype tt, N_Vector cc, N_Vector cp, N_Vector res,
                  void* user_data)
{
  sunindextype jx, jy, is, yloc, loc, np;
  sunrealtype *ccv, *cpv, *resv;
  UserData webdata = (UserData)user_data;
  np = webdata->np;

  /* Pull from device to host */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  N_VCopyFromDevice_Cuda(cc);
  N_VCopyFromDevice_Cuda(cp);
  ccv  = N_VGetHostArrayPointer_Cuda(cc);
  cpv  = N_VGetHostArrayPointer_Cuda(cp);
  resv = N_VGetHostArrayPointer_Cuda(res);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  N_VCopyFromDevice_Hip(cc);
  N_VCopyFromDevice_Hip(cp);
  ccv  = N_VGetHostArrayPointer_Hip(cc);
  cpv  = N_VGetHostArrayPointer_Hip(cp);
  resv = N_VGetHostArrayPointer_Hip(res);
#else
  ccv  = N_VGetArrayPointer(cc);
  cpv  = N_VGetArrayPointer(cp);
  resv = N_VGetArrayPointer(res);
#endif

  /* Get host pointer for rates vector */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  sunrealtype* ratesv = N_VGetHostArrayPointer_Cuda(webdata->rates);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  sunrealtype* ratesv = N_VGetHostArrayPointer_Hip(webdata->rates);
#else
  sunrealtype* ratesv = N_VGetArrayPointer(webdata->rates);
#endif

  /* Compute Fweb into resv (right-hand sides) on host */
  Fweb(tt, ccv, resv, webdata);

  /* Convert F to residual form:
     prey:     res[is] = cp[is] - F[is]
     predator: res[is] = -F[is]           */
  for (jy = 0; jy < MY; jy++)
  {
    yloc = NSMX * jy;
    for (jx = 0; jx < MX; jx++)
    {
      loc = yloc + NUM_SPECIES * jx;
      for (is = 0; is < NUM_SPECIES; is++)
      {
        if (is < np) { resv[loc + is] = cpv[loc + is] - resv[loc + is]; }
        else         { resv[loc + is] = -resv[loc + is]; }
      }
    }
  }

  /* Push result back to device */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  N_VCopyToDevice_Cuda(res);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  N_VCopyToDevice_Hip(res);
#endif

  return 0;
}

// Set up the sparsity structure of the matrix
static int jacFoodInit(SUNMatrix JJ, UserData webdata)
{
  sunindextype ns = webdata->ns, mx = webdata->mx, my = webdata->my;
  sunindextype k, node, is, js, jx, jy, sum = 0, entry = 0;

  // Get pointers
  sunindextype* rowptrs = SUNMatrix_ReSolve_IndexPointers(JJ, ReSolve::memory::HOST);
  sunindextype* colvals = SUNMatrix_ReSolve_IndexValues(JJ, ReSolve::memory::HOST);
  sunrealtype*  data    = SUNMatrix_ReSolve_Data(JJ, ReSolve::memory::HOST);

  rowptrs[0] = 0;

  // Fill the matrix
  for (k = 0; k < NEQ; k++)
  {
    node = k / ns;
    is   = k % ns;
    jx   = node % mx;
    jy   = node / mx;

    sunbooleantype has_left  = (jx > 0);
    sunbooleantype has_right = (jx < mx - 1);
    sunbooleantype has_down  = (jy > 0);
    sunbooleantype has_up    = (jy < my - 1);

    sunindextype node_left  = jy * mx + (jx - 1);
    sunindextype node_right = jy * mx + (jx + 1);
    sunindextype node_down  = (jy - 1) * mx + jx;
    sunindextype node_up    = (jy + 1) * mx + jx;

    // down neighbor
    if (has_down)
    {
      colvals[entry] = node_down * ns + is;
      data[entry]    = ZERO;
      sum++; entry++;
    }
    // left neighbor
    if (has_left)
    {
      colvals[entry] = node_left * ns + is;
      data[entry]    = ZERO;
      sum++; entry++;
    }
    // self block (dense, ns entries)
    for (js = 0; js < ns; js++)
    {
      colvals[entry] = node * ns + js;
      data[entry]    = ZERO;
      sum++; entry++;
    }
    // right neighbor
    if (has_right)
    {
      colvals[entry] = node_right * ns + is;
      data[entry]    = ZERO;
      sum++; entry++;
    }
    // up neighbor
    if (has_up)
    {
      colvals[entry] = node_up * ns + is;
      data[entry]    = ZERO;
      sum++; entry++;
    }

    rowptrs[k + 1] = sum;
  }

  // Set to updated
  SUNMatrix_ReSolve_SetUpdated(JJ, ReSolve::memory::HOST);

  ReSolve::memory::MemorySpace memspace = SUNMatrix_ReSolve_MemorySpace(JJ);
  if (memspace != ReSolve::memory::HOST)
  {
    SUNMatrix_ReSolve_SyncData(JJ, memspace);
  }

  return (0);
}

static int jacFoodWeb(sunrealtype tt, sunrealtype cj, N_Vector cc, N_Vector cp,
               N_Vector resvec, SUNMatrix JJ, void* user_data,
               N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
{
  UserData webdata = (UserData)user_data;
  sunindextype ns = webdata->ns, mx = webdata->mx, my = webdata->my, np = webdata->np;
  sunindextype k, node, is, js, jx, jy;

  sunrealtype* data     = SUNMatrix_ReSolve_Data(JJ, ReSolve::memory::HOST);
  sunindextype* rowptrs = SUNMatrix_ReSolve_IndexPointers(JJ, ReSolve::memory::HOST);
  sunrealtype* cdata    = N_VGetArrayPointer(cc);  // (device variants as before)

  for (k = 0; k < NEQ; k++)          // k = global row index, same style as jacHeat's k
  {
    node = k / ns;                   // which grid node
    is   = k % ns;                   // which species — new vs. jacHeat, since ns>1 here
    jx   = node % mx;
    jy   = node / mx;

    sunbooleantype has_left  = (jx > 0);
    sunbooleantype has_right = (jx < mx - 1);
    sunbooleantype has_down  = (jy > 0);
    sunbooleantype has_up    = (jy < my - 1);

    sunrealtype* cxy = IJ_Vptr_h(cdata, jx, jy);
    sunrealtype xx = jx * webdata->dx, yy = jy * webdata->dy;
    sunrealtype fac = ONE + ALPHA*xx*yy + BETA*sin(FOURPI*xx)*sin(FOURPI*yy);

    sunrealtype rowsum = ZERO;
    for (js = 0; js < ns; js++) rowsum += acoef[is][js]*cxy[js];

    sunindextype entry = rowptrs[k];   // same idea as jacHeat: reuse precomputed offset

    if (has_down) { data[entry] = -coy[is]; entry++; }
    if (has_left) { data[entry] = -cox[is]; entry++; }

    for (js = 0; js < ns; js++)
    {
      sunrealtype val = -cxy[is]*acoef[is][js];
      if (is == js)
      {
        val -= bcoef[is]*fac + rowsum;
        sunrealtype diff_diag = SUN_RCONST(2.0)*cox[is] + SUN_RCONST(2.0)*coy[is];
        if (!has_left)  diff_diag -= cox[is];
        if (!has_right) diff_diag -= cox[is];
        if (!has_down)  diff_diag -= coy[is];
        if (!has_up)    diff_diag -= coy[is];
        val += diff_diag;
        if (is < np) val += cj;
      }
      data[entry] = val; entry++;
    }

    if (has_right) { data[entry] = -cox[is]; entry++; }
    if (has_up)    { data[entry] = -coy[is]; entry++; }
  }

  SUNMatrix_ReSolve_SetUpdated(JJ, ReSolve::memory::HOST);
  ReSolve::memory::MemorySpace memspace = SUNMatrix_ReSolve_MemorySpace(JJ);
  if (memspace != ReSolve::memory::HOST) 
  {
    SUNMatrix_ReSolve_SyncData(JJ, memspace);
  }
  return 0;
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

static void InitUserData(UserData webdata)
{
  sunindextype i, j, np;
  sunrealtype *a1, *a2, *a3, *a4, dx2, dy2;

  webdata->mx  = MX;
  webdata->my  = MY;
  webdata->ns  = NUM_SPECIES;
  webdata->np  = NPREY;
  webdata->dx  = AX / (MX - 1);
  webdata->dy  = AY / (MY - 1);
  webdata->Neq = NEQ;

  np  = webdata->np;
  dx2 = webdata->dx * webdata->dx;
  dy2 = webdata->dy * webdata->dy;

  for (i = 0; i < np; i++)
  {
    a1 = &(acoef[i][np]);
    a2 = &(acoef[i + np][0]);
    a3 = &(acoef[i][0]);
    a4 = &(acoef[i + np][np]);
    for (j = 0; j < np; j++)
    {
      *a1++ = -GG;
      *a2++ = EE;
      *a3++ = ZERO;
      *a4++ = ZERO;
    }
    acoef[i][i]           = -AA;
    acoef[i + np][i + np] = -AA;

    bcoef[i]      = BB;
    bcoef[i + np] = -BB;
    cox[i]        = DPREY / dx2;
    cox[i + np]   = DPRED / dx2;
    coy[i]        = DPREY / dy2;
    coy[i + np]   = DPRED / dy2;
  }
}

/*
 * SetInitialProfiles: works entirely on host arrays, syncs to device
 * at the end. Mirrors the pattern from idaHeat2D_ReSolve.
 */
static void SetInitialProfiles(N_Vector cc, N_Vector cp, N_Vector id,
                               UserData webdata)
{
  sunindextype loc, yloc, is, jx, jy, np;
  sunrealtype xx, yy, xyfactor;
  sunrealtype *ccv, *cpv, *idv;

  np = webdata->np;

  /* Get host pointers */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  ccv = N_VGetHostArrayPointer_Cuda(cc);
  cpv = N_VGetHostArrayPointer_Cuda(cp);
  idv = N_VGetHostArrayPointer_Cuda(id);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  ccv = N_VGetHostArrayPointer_Hip(cc);
  cpv = N_VGetHostArrayPointer_Hip(cp);
  idv = N_VGetHostArrayPointer_Hip(id);
#else
  ccv = N_VGetArrayPointer(cc);
  cpv = N_VGetArrayPointer(cp);
  idv = N_VGetArrayPointer(id);
#endif

  /* Initialize id to 1 everywhere on host, will be corrected for
     predators below */
  for (jy = 0; jy < MY; jy++)
  {
    yloc = NSMX * jy;
    for (jx = 0; jx < MX; jx++)
    {
      loc = yloc + NUM_SPECIES * jx;
      for (is = 0; is < NUM_SPECIES; is++) { idv[loc + is] = ONE; }
    }
  }

  /* Initialize cc and id on all grid points */
  for (jy = 0; jy < MY; jy++)
  {
    yy   = jy * webdata->dy;
    yloc = NSMX * jy;
    for (jx = 0; jx < MX; jx++)
    {
      xx       = jx * webdata->dx;
      xyfactor = SUN_RCONST(16.0) * xx * (ONE - xx) * yy * (ONE - yy);
      xyfactor *= xyfactor;
      loc = yloc + NUM_SPECIES * jx;

      for (is = 0; is < NUM_SPECIES; is++)
      {
        if (is < np)
        {
          ccv[loc + is] = SUN_RCONST(10.0) + (sunrealtype)(is + 1) * xyfactor;
          idv[loc + is] = ONE;
        }
        else
        {
          ccv[loc + is] = SUN_RCONST(1.0e5);
          idv[loc + is] = ZERO;
        }
      }
    }
  }

  /* Zero cp on host before computing prey derivatives */
  for (sunindextype k = 0; k < NEQ; k++) { cpv[k] = ZERO; }

  /* Sync cc, cp, id to device before calling resweb via Fweb */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  N_VCopyToDevice_Cuda(cc);
  N_VCopyToDevice_Cuda(cp);
  N_VCopyToDevice_Cuda(id);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  N_VCopyToDevice_Hip(cc);
  N_VCopyToDevice_Hip(cp);
  N_VCopyToDevice_Hip(id);
#endif

  /* Set cp for prey by calling Fweb on host data directly */
  Fweb(ZERO, ccv, cpv, webdata);

  /* Zero cp for predators */
  for (jy = 0; jy < MY; jy++)
  {
    yloc = NSMX * jy;
    for (jx = 0; jx < MX; jx++)
    {
      loc = yloc + NUM_SPECIES * jx;
      for (is = np; is < NUM_SPECIES; is++) { cpv[loc + is] = ZERO; }
    }
  }

  /* Sync final cc, cp, id to device */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  N_VCopyToDevice_Cuda(cc);
  N_VCopyToDevice_Cuda(cp);
  N_VCopyToDevice_Cuda(id);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  N_VCopyToDevice_Hip(cc);
  N_VCopyToDevice_Hip(cp);
  N_VCopyToDevice_Hip(id);
#endif
}

/*
 * Fweb: operates purely on raw host pointers, not N_Vectors.
 * This avoids needing to pull/push inside Fweb itself, letting
 * the callers (resweb, SetInitialProfiles) manage GPU transfers.
 */
static void Fweb(sunrealtype tcalc, sunrealtype* ccv, sunrealtype* cratev,
                 UserData webdata)
{
  sunindextype jx, jy, is, idyu, idyl, idxu, idxl;
  sunrealtype xx, yy, dcyli, dcyui, dcxli, dcxui;

#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  sunrealtype* ratesv = N_VGetHostArrayPointer_Cuda(webdata->rates);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  sunrealtype* ratesv = N_VGetHostArrayPointer_Hip(webdata->rates);
#else
  sunrealtype* ratesv = N_VGetArrayPointer(webdata->rates);
#endif

  for (jy = 0; jy < MY; jy++)
  {
    yy   = webdata->dy * jy;
    idyu = (jy != MY - 1) ? NSMX : -NSMX;
    idyl = (jy != 0)      ? NSMX : -NSMX;

    for (jx = 0; jx < MX; jx++)
    {
      xx   = webdata->dx * jx;
      idxu = (jx != MX - 1) ? NUM_SPECIES : -NUM_SPECIES;
      idxl = (jx != 0)      ? NUM_SPECIES : -NUM_SPECIES;

      sunrealtype* cxy     = IJ_Vptr_h(ccv,    jx, jy);
      sunrealtype* ratesxy = IJ_Vptr_h(ratesv,  jx, jy);
      sunrealtype* cratexy = IJ_Vptr_h(cratev,  jx, jy);

      WebRates(xx, yy, cxy, ratesxy, webdata);

      for (is = 0; is < NUM_SPECIES; is++)
      {
        dcyli = *(cxy + is) - *(cxy - idyl + is);
        dcyui = *(cxy + idyu + is) - *(cxy + is);
        dcxli = *(cxy + is) - *(cxy - idxl + is);
        dcxui = *(cxy + idxu + is) - *(cxy + is);

        cratexy[is] = coy[is] * (dcyui - dcyli) +
                      cox[is] * (dcxui - dcxli) +
                      ratesxy[is];
      }
    }
  }
}

static void WebRates(sunrealtype xx, sunrealtype yy, sunrealtype* cxy,
                     sunrealtype* ratesxy, UserData webdata)
{
  int is;
  sunrealtype fac;

  for (is = 0; is < NUM_SPECIES; is++)
  {
    ratesxy[is] = dotprod(NUM_SPECIES, cxy, acoef[is]);
  }

  fac = ONE + ALPHA * xx * yy + BETA * sin(FOURPI * xx) * sin(FOURPI * yy);

  for (is = 0; is < NUM_SPECIES; is++)
  {
    ratesxy[is] = cxy[is] * (bcoef[is] * fac + ratesxy[is]);
  }
}

static sunrealtype dotprod(sunindextype size, sunrealtype* x1, sunrealtype* x2)
{
  sunindextype i;
  sunrealtype temp = ZERO;
  for (i = 0; i < size; i++) { temp += x1[i] * x2[i]; }
  return temp;
}

static void PrintHeader(sunrealtype rtol, sunrealtype atol,
                        std::string hwbackend)
{
  printf("\nidaFoodWeb_ReSolve: Predator-prey DAE example for IDA\n\n");
  printf("Number of species ns: %d", NUM_SPECIES);
  printf("     Mesh dimensions: %d x %d", MX, MY);
  printf("     System size: %d\n", NEQ);
#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("Tolerance parameters:  rtol = %Lg   atol = %Lg\n", rtol, atol);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("Tolerance parameters:  rtol = %g   atol = %g\n", rtol, atol);
#else
  printf("Tolerance parameters:  rtol = %g   atol = %g\n", rtol, atol);
#endif
  std::cout << "Linear solver: ReSolve KLU with " << hwbackend << " backend\n";
  printf("CalcIC called to correct initial predator concentrations.\n\n");
  printf("-----------------------------------------------------------\n");
  printf("  t        bottom-left  top-right");
  printf("    | nst  k      h\n");
  printf("-----------------------------------------------------------\n\n");
}

static void PrintOutput(void* mem, N_Vector c, sunrealtype t)
{
  int i, kused, retval;
  long int nst;
  sunrealtype *c_bl, *c_tr, hused;

  /* Pull to host for printing */
#if defined(SUNDIALS_RESOLVE_BACKENDS_CUDA)
  N_VCopyFromDevice_Cuda(c);
  sunrealtype* cv = N_VGetHostArrayPointer_Cuda(c);
#elif defined(SUNDIALS_RESOLVE_BACKENDS_HIP)
  N_VCopyFromDevice_Hip(c);
  sunrealtype* cv = N_VGetHostArrayPointer_Hip(c);
#else
  sunrealtype* cv = N_VGetArrayPointer(c);
#endif

  retval = IDAGetLastOrder(mem, &kused);
  check_retval(&retval, "IDAGetLastOrder", 1);
  retval = IDAGetNumSteps(mem, &nst);
  check_retval(&retval, "IDAGetNumSteps", 1);
  retval = IDAGetLastStep(mem, &hused);
  check_retval(&retval, "IDAGetLastStep", 1);

  c_bl = IJ_Vptr_h(cv, 0, 0);
  c_tr = IJ_Vptr_h(cv, MX - 1, MY - 1);

#if defined(SUNDIALS_DOUBLE_PRECISION)
  printf("%8.2e %12.4e %12.4e   | %3ld  %1d %12.4e\n", t, c_bl[0], c_tr[0],
         nst, kused, hused);
  for (i = 1; i < NUM_SPECIES; i++)
    printf("         %12.4e %12.4e   |\n", c_bl[i], c_tr[i]);
#else
  printf("%8.2e %12.4e %12.4e   | %3ld  %1d %12.4e\n", t, c_bl[0], c_tr[0],
         nst, kused, hused);
  for (i = 1; i < NUM_SPECIES; i++)
    printf("         %12.4e %12.4e   |\n", c_bl[i], c_tr[i]);
#endif
  printf("\n");
}

static void PrintFinalStats(void* mem)
{
  long int nst, nre, nreLS, nni, nnf, nje, netf, ncfn;
  int retval;

  retval = IDAGetNumSteps(mem, &nst);
  check_retval(&retval, "IDAGetNumSteps", 1);
  retval = IDAGetNumNonlinSolvIters(mem, &nni);
  check_retval(&retval, "IDAGetNumNonlinSolvIters", 1);
  retval = IDAGetNumResEvals(mem, &nre);
  check_retval(&retval, "IDAGetNumResEvals", 1);
  retval = IDAGetNumErrTestFails(mem, &netf);
  check_retval(&retval, "IDAGetNumErrTestFails", 1);
  retval = IDAGetNumNonlinSolvConvFails(mem, &nnf);
  check_retval(&retval, "IDAGetNumNonlinSolvConvFails", 1);
  retval = IDAGetNumStepSolveFails(mem, &ncfn);
  check_retval(&retval, "IDAGetNumStepSolveFails", 1);
  retval = IDAGetNumJacEvals(mem, &nje);
  check_retval(&retval, "IDAGetNumJacEvals", 1);
  retval = IDAGetNumLinResEvals(mem, &nreLS);
  check_retval(&retval, "IDAGetNumLinResEvals", 1);

  printf("-----------------------------------------------------------\n");
  printf("Final run statistics: \n\n");
  printf("Number of steps                    = %ld\n", nst);
  printf("Number of residual evaluations     = %ld\n", nre + nreLS);
  printf("Number of Jacobian evaluations     = %ld\n", nje);
  printf("Number of nonlinear iterations     = %ld\n", nni);
  printf("Number of error test failures      = %ld\n", netf);
  printf("Number of nonlinear conv. failures = %ld\n", nnf);
  printf("Number of step solver failures     = %ld\n", ncfn);
}

static int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }
  else if (opt == 1)
  {
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return 1;
    }
  }
  else if (opt == 2 && returnvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }
  return 0;
}