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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_resolve.hpp>
#include <sunlinsol/sunlinsol_resolve.hpp>

#include <resolve/SystemSolver.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include <nvector/nvector_cuda.h>
#include <sunmemory/sunmemory_cuda.h>

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

#define IJ_Vptr_h(ptr, i, j) ((ptr) + (i) * NUM_SPECIES + (j) * NSMX)

/* CUDA constant memory: run-fixed problem data, filled once in InitUserData */
__constant__ sunrealtype d_acoef[NUM_SPECIES][NUM_SPECIES];
__constant__ sunrealtype d_bcoef[NUM_SPECIES];
__constant__ sunrealtype d_cox[NUM_SPECIES];
__constant__ sunrealtype d_coy[NUM_SPECIES];

typedef struct
{
  sunindextype Neq, ns, np, mx, my;
  sunrealtype dx, dy;
  sunrealtype cox[NUM_SPECIES], coy[NUM_SPECIES], bcoef[NUM_SPECIES], acoef[NUM_SPECIES][NUM_SPECIES];
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
static void Fweb(sunrealtype tcalc, N_Vector ccv, N_Vector cratev,
                 UserData webdata);
static int check_retval(void* returnvalue, const char* funcname, int opt);

/*
 *--------------------------------------------------------------------
 * CUDA device helper functions
 *--------------------------------------------------------------------
 */

__device__ sunrealtype dotprod(sunindextype size, const sunrealtype* x1,
                                const sunrealtype* x2)
{
  sunrealtype temp = ZERO;
  for (sunindextype i = 0; i < size; i++) { temp += x1[i] * x2[i]; }
  return temp;
}

__device__ void WebRates(sunrealtype xx, sunrealtype yy,
                          const sunrealtype* cxy, sunrealtype* ratesxy)
{
  int is;
  sunrealtype fac;

  for (is = 0; is < NUM_SPECIES; is++)
  {
    ratesxy[is] = dotprod(NUM_SPECIES, cxy, d_acoef[is]);
  }

  fac = ONE + ALPHA * xx * yy + BETA * sin(FOURPI * xx) * sin(FOURPI * yy);

  for (is = 0; is < NUM_SPECIES; is++)
  {
    ratesxy[is] = cxy[is] * (d_bcoef[is] * fac + ratesxy[is]);
  }
}

/*
 *--------------------------------------------------------------------
 * CUDA Kernels
 *--------------------------------------------------------------------
 */

__global__ void fWebKernel(sunrealtype tcalc, const sunrealtype* cc_data,
                           sunrealtype* crate_data, sunrealtype* rates_data,
                           sunrealtype dx, sunrealtype dy)
{
  sunindextype jx, jy, tid, is, idyu, idyl, idxu, idxl;
  sunrealtype xx, yy, dcyli, dcyui, dcxli, dcxui;
  const sunrealtype* cxy;
  sunrealtype *ratesxy, *cratexy;

  tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < MX * MY)
  {
    jx = tid % MX;
    jy = tid / MX;

    yy   = dy * jy;
    idyu = (jy != MY - 1) ? NSMX : -NSMX;
    idyl = (jy != 0) ? NSMX : -NSMX;

    xx   = dx * jx;
    idxu = (jx != MX - 1) ? NUM_SPECIES : -NUM_SPECIES;
    idxl = (jx != 0) ? NUM_SPECIES : -NUM_SPECIES;

    cxy     = cc_data + jx * NUM_SPECIES + jy * NSMX;
    ratesxy = rates_data + jx * NUM_SPECIES + jy * NSMX;
    cratexy = crate_data + jx * NUM_SPECIES + jy * NSMX;

    /* Get interaction vector at this grid point. */
    WebRates(xx, yy, cxy, ratesxy);

    /* Loop over species, do differencing, load crate segment. */
    for (is = 0; is < NUM_SPECIES; is++)
    {
      /* Differencing in y. */
      dcyli = *(cxy + is) - *(cxy - idyl + is);
      dcyui = *(cxy + idyu + is) - *(cxy + is);

      /* Differencing in x. */
      dcxli = *(cxy + is) - *(cxy - idxl + is);
      dcxui = *(cxy + idxu + is) - *(cxy + is);

      /* Compute the crate values at (xx,yy). */
      cratexy[is] = d_coy[is] * (dcyui - dcyli) + d_cox[is] * (dcxui - dcxli) +
                    ratesxy[is];
    }
  }
}

__global__ void resWebKernel(sunrealtype* resv, const sunrealtype* cpv,
                             sunindextype np)
{
  sunindextype tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < MX * MY)
  {
    sunindextype jx  = tid % MX;
    sunindextype jy  = tid / MX;
    sunindextype loc = jx * NUM_SPECIES + jy * NSMX;

    for (int is = 0; is < NUM_SPECIES; is++)
    {
      sunindextype idx = loc + is;
      if (is < np) { resv[idx] = cpv[idx] - resv[idx]; }
      else { resv[idx] = -resv[idx]; }
    }
  }
}

__global__ void jacFoodWebKernel(sunrealtype cj, const sunrealtype* cc_data,
                                  sunrealtype* data, const sunindextype* rowptrs,
                                  sunrealtype dx, sunrealtype dy, sunindextype np)
{
  sunindextype tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < MX * MY)
  {
    sunindextype jx = tid % MX;
    sunindextype jy = tid / MX;

    sunbooleantype has_left  = (jx > 0);
    sunbooleantype has_right = (jx < MX - 1);
    sunbooleantype has_down  = (jy > 0);
    sunbooleantype has_up    = (jy < MY - 1);

    const sunrealtype* cxy = cc_data + jx * NUM_SPECIES + jy * NSMX;
    sunrealtype xx  = jx * dx, yy = jy * dy;
    sunrealtype fac = ONE + ALPHA * xx * yy + BETA * sin(FOURPI * xx) * sin(FOURPI * yy);

    for (int is = 0; is < NUM_SPECIES; is++)
    {
      sunindextype k = tid * NUM_SPECIES + is;

      sunrealtype rowsum = ZERO;
      for (int js = 0; js < NUM_SPECIES; js++)
      {
        rowsum += d_acoef[is][js] * cxy[js];
      }

      sunindextype entry = rowptrs[k];

      if (has_down) { data[entry] = -d_coy[is]; entry++; }
      if (has_left) { data[entry] = -d_cox[is]; entry++; }

      for (int js = 0; js < NUM_SPECIES; js++)
      {
        sunrealtype val = -cxy[is] * d_acoef[is][js];
        if (is == js)
        {
          val -= d_bcoef[is] * fac + rowsum;
          sunrealtype diff_diag = SUN_RCONST(2.0) * d_cox[is] +
                                   SUN_RCONST(2.0) * d_coy[is];
          if (!has_left)  diff_diag -= d_cox[is];
          if (!has_right) diff_diag -= d_cox[is];
          if (!has_down)  diff_diag -= d_coy[is];
          if (!has_up)    diff_diag -= d_coy[is];
          val += diff_diag;
          if (is < np) val += cj;
        }
        data[entry] = val; entry++;
      }

      if (has_right) { data[entry] = -d_cox[is]; entry++; }
      if (has_up)    { data[entry] = -d_coy[is]; entry++; }
    }
  }
}

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

  /* Allocate vectors */
  cc = N_VNew_Cuda(NEQ, ctx);
  if (check_retval((void*)cc, "N_VNew_Cuda", 0)) { return 1; }
  cp = N_VClone(cc);
  if (check_retval((void*)cp, "N_VClone", 0)) { return 1; }
  id = N_VClone(cc);
  if (check_retval((void*)id, "N_VClone", 0)) { return 1; }

  /* Allocate and initialize user data */
  webdata        = (UserData)malloc(sizeof *webdata);
  webdata->rates = N_VClone(cc);
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
  ReSolve::memory::MemorySpace memspace = ReSolve::memory::DEVICE;

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

  ReSolve::LinAlgWorkspaceCUDA workspace;
  std::string refactor = "cusolverrf";
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

  PrintHeader(rtol, atol, "CUDA");
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
  UserData webdata = (UserData)user_data;
  sunindextype np  = webdata->np;

  /* Call Fweb to set res to vector of right-hand sides (device). */
  Fweb(tt, cc, res, webdata);

  const sunrealtype* cp_data = N_VGetDeviceArrayPointer_Cuda(cp);
  sunrealtype* res_data      = N_VGetDeviceArrayPointer_Cuda(res);

  unsigned block = 256;
  unsigned grid  = (MX * MY + block - 1) / block;

  resWebKernel<<<grid, block>>>(res_data, cp_data, np);

  return (0);
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

  const sunrealtype* cc_data = N_VGetDeviceArrayPointer_Cuda(cc);
  sunrealtype* data          = SUNMatrix_ReSolve_Data(JJ, ReSolve::memory::DEVICE);
  sunindextype* rowptrs      = SUNMatrix_ReSolve_IndexPointers(JJ, ReSolve::memory::DEVICE);

  unsigned block = 256;
  unsigned grid  = (MX * MY + block - 1) / block;

  jacFoodWebKernel<<<grid, block>>>(cj, cc_data, data, rowptrs,
                                     webdata->dx, webdata->dy, webdata->np);
  cudaDeviceSynchronize();   // <-- add this
  SUNMatrix_ReSolve_SetUpdated(JJ, ReSolve::memory::DEVICE);
  SUNMatrix_ReSolve_SyncData(JJ, ReSolve::memory::HOST);

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

  /* Send the arrays to the device */
  cudaMemcpyToSymbol(d_acoef, acoef,
                     NUM_SPECIES * NUM_SPECIES * sizeof(sunrealtype));
  cudaMemcpyToSymbol(d_bcoef, bcoef, NUM_SPECIES * sizeof(sunrealtype));
  cudaMemcpyToSymbol(d_cox, cox, NUM_SPECIES * sizeof(sunrealtype));
  cudaMemcpyToSymbol(d_coy, coy, NUM_SPECIES * sizeof(sunrealtype));
}

/*
 * SetInitialProfiles: Set initial conditions in cc, cp, and id.
 * A polynomial profile is used for the prey cc values, and a constant
 * (1.0e5) is loaded as the initial guess for the predator cc values.
 * The id values are set to 1 for the prey and 0 for the predators.
 * The prey cp values are set according to the given system (via Fweb,
 * device), and the predator cp values are set to zero.
 */

static void SetInitialProfiles(N_Vector cc, N_Vector cp, N_Vector id,
                               UserData webdata)
{
  sunindextype loc, yloc, is, jx, jy, np;
  sunrealtype xx, yy, xyfactor;
  sunrealtype *ccv, *idv, *cpv;

  ccv = N_VGetHostArrayPointer_Cuda(cc);
  idv = N_VGetHostArrayPointer_Cuda(id);
  np  = webdata->np;

  /* Loop over grid, load cc values and id values (host arrays). */
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

  N_VCopyToDevice_Cuda(cc);
  N_VCopyToDevice_Cuda(id);

  /* Set c' for the prey by calling Fweb (device kernel writes into cp). */
  Fweb(ZERO, cc, cp, webdata);

  /* Set c' for predators to 0. */
  N_VCopyFromDevice_Cuda(cp);
  cpv = N_VGetHostArrayPointer_Cuda(cp);

  for (jy = 0; jy < MY; jy++)
  {
    yloc = NSMX * jy;
    for (jx = 0; jx < MX; jx++)
    {
      loc = yloc + NUM_SPECIES * jx;
      for (is = np; is < NUM_SPECIES; is++) { cpv[loc + is] = ZERO; }
    }
  }

  N_VCopyToDevice_Cuda(cp);
}

/*
 * Fweb: host wrapper -- extracts device pointers and launches fWebKernel.
 */
static void Fweb(sunrealtype tcalc, N_Vector cc, N_Vector crate, UserData webdata)
{
  const sunrealtype* cc_data = N_VGetDeviceArrayPointer_Cuda(cc);
  sunrealtype* crate_data    = N_VGetDeviceArrayPointer_Cuda(crate);
  sunrealtype* rates_data    = N_VGetDeviceArrayPointer_Cuda(webdata->rates);

  unsigned block = 256;
  unsigned grid  = (MX * MY + block - 1) / block;

  fWebKernel<<<grid, block>>>(tcalc, cc_data, crate_data, rates_data,
                              webdata->dx, webdata->dy);
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
  N_VCopyFromDevice_Cuda(c);
  sunrealtype* cv = N_VGetHostArrayPointer_Cuda(c);

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