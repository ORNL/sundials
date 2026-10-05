/* -----------------------------------------------------------------
 * Programmer(s): Jeffery Zhang
 * -----------------------------------------------------------------
 * Based on work by Slaven Peles, Allan Taylor, Alan Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Example problem for IDA: 2D heat equation, serial, GMRES.
 *
 * This example solves a discretized 2D heat equation problem.
 * This version uses the Krylov solver Spgmr.
 *
 * The DAE system solved is a spatial discretization of the PDE
 *          du/dt = d^2u/dx^2 + d^2u/dy^2
 * on the unit square. The boundary condition is u = 0 on all edges.
 * Initial conditions are given by u = 16 x (1 - x) y (1 - y). The
 * PDE is treated with central differences on a uniform M x M grid.
 * The values of u at the interior points satisfy ODEs, and
 * equations u = 0 at the boundaries are appended, to form a DAE
 * system of size N = M^2. Here M = 10.
 *
 * The system is solved with IDA using the Krylov linear solver
 * SPGMR. The preconditioner uses the diagonal elements of the
 * Jacobian only. Routines for preconditioning, required by
 * SPGMR, are supplied here. The constraints u >= 0 are posed
 * for all components. Output is taken at t = 0, .01, .02, .04,
 * ..., 10.24. Two cases are run -- with the Gram-Schmidt type
 * being Modified in the first case, and Classical in the second.
 * The second run uses IDAReInit.
 * -----------------------------------------------------------------*/

#include <ida/ida.h> /* prototypes for IDA methods            */
#include <math.h>
#include <nvector/nvector_cuda.h> /* access to CUDA N_Vector               */
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_types.h> /* definition of type sunrealtype           */
#include <sunmatrix/sunmatrix_resolve.hpp>  /* access to ReSolve SUNMatrix          */
#include <sunlinsol/sunlinsol_resolve.hpp>  /* access to ReSolve Linear Solver      */

// ReSolve headers
#include <resolve/SystemSolver.hpp>       
#include <resolve/LinSolverIterative.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

/* Problem Constants */

#define NOUT  11
#define MGRID 10
#define NEQ   MGRID* MGRID
#define ZERO  SUN_RCONST(0.0)
#define ONE   SUN_RCONST(1.0)
#define TWO   SUN_RCONST(2.0)
#define FOUR  SUN_RCONST(4.0)

/* User data type */

struct _UserData
{
  sunindextype mm;  /* number of grid points in one dimension */
  sunindextype neq; /* number of equations */
  sunrealtype dx;
  sunrealtype coeff;
  N_Vector pp; /* vector of prec. diag. elements */
};

typedef _UserData* UserData;

/* Prototypes for functions called by IDA */

int resHeat(sunrealtype tres, N_Vector uu, N_Vector up, N_Vector resval,
            void* user_data);

int jacHeatInit(SUNMatrix JJ);

int jacHeat(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp,
            N_Vector resvec, SUNMatrix JJ, void* user_data, N_Vector tempv1,
            N_Vector tempv2, N_Vector tempv3);

/* Prototypes for private functions */

static int SetInitialProfile(UserData data, N_Vector uu, N_Vector up,
                             N_Vector id,  N_Vector res);
static void PrintHeader(sunrealtype rtol, sunrealtype atol);
static void PrintOutput(void* mem, sunrealtype t, N_Vector uu);
static int check_flag(void* flagvalue, const char* funcname, int opt);

/*
 *--------------------------------------------------------------------
 * CUDA Kernels
 *--------------------------------------------------------------------
 */

__global__ void resHeatKernel(const sunrealtype* uu, const sunrealtype* up,
                              sunrealtype* rr, sunindextype mm, sunrealtype coeff)
{
  sunindextype i, j, tid;

  /* Loop over all grid points. */
  tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < mm * mm)
  {
    i = tid % mm;
    j = tid / mm;

    if (j == 0 || j == mm - 1 || i == 0 || i == mm - 1)
    {
      /* Initialize rr to uu, to take care of boundary equations. */
      rr[tid] = uu[tid];
    }
    else
    {
      /* Loop over interior points; set res = up - (central difference). */
      sunrealtype dif1 = uu[tid - 1] + uu[tid + 1] - TWO * uu[tid];
      sunrealtype dif2 = uu[tid - mm] + uu[tid + mm] - TWO * uu[tid];
      rr[tid]          = up[tid] - coeff * (dif1 + dif2);
    }
  }
}

__global__ void jacHeatKernel(sunrealtype* data, sunindextype* rowptrs,
                                sunrealtype c_j)
{
  sunrealtype dx   = ONE / (MGRID - ONE);
  sunrealtype beta = SUN_RCONST(4.0) / (dx * dx) + c_j;
  sunindextype i, j, tid, entry;

  /* Loop over all grid points. */
  tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < MGRID * MGRID)
  {
    i = tid % MGRID;
    j = tid / MGRID;

    entry = rowptrs[tid];

    if (j == 0 || j == MGRID - 1 || i == 0 || i == MGRID - 1)
    {
      data[entry] = ONE;
    }
    else
    {
      data[entry] = -ONE/(dx*dx);
      data[entry + 1] = -ONE/(dx*dx);
      data[entry + 2] = beta;
      data[entry + 3] = -ONE/(dx*dx);
      data[entry + 4] = -ONE/(dx*dx);
    }
  }
}

__global__ void setInitHeatKernel(sunrealtype* up, sunrealtype* id, sunindextype mm)
{
  sunindextype i, j, tid;

  /* Loop over all grid points. */
  tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < mm * mm)
  {
    i = tid % mm;
    j = tid / mm;

    if (j == 0 || j == mm - 1 || i == 0 || i == mm - 1) 
    { 
      up[tid] = ZERO; 
      id[tid] = ZERO;
    }
    else
    {
      id[tid] = ONE;
    }
  }
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

int main(int argc, char* argv[])
{
  void* mem;
  UserData data;
  N_Vector uu, up, constraints, id, res;
  int ier, iout;
  sunrealtype rtol, atol, t0, t1, tout, tret;
  sunindextype nnz;
  long int netf, ncfn;
  SUNLinearSolver LS;
  SUNContext ctx;
  SUNMatrix A;

  mem  = NULL;
  data = NULL;
  uu = up = constraints = res = NULL;
  LS                          = NULL;

  /* Create the SUNDIALS context object for this simulation */

  ier = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_flag(&ier, "SUNContext_Create", 1)) { return 1; }

  /* Assign parameters in the user data structure. */

  data     = (UserData)malloc(sizeof *data);
  data->pp = NULL;
  if (check_flag((void*)data, "malloc", 2)) { return (1); }

  data->mm    = MGRID;
  data->neq   = data->mm * data->mm;
  data->dx    = ONE / (data->mm - ONE);
  data->coeff = ONE / (data->dx * data->dx);

  /* Allocate N-vectors and the user data structure objects. */

  uu = N_VNew_Cuda(data->neq, ctx);
  if (check_flag((void*)uu, "N_VNew_Serial", 0)) { return (1); }

  up = N_VClone(uu);
  if (check_flag((void*)up, "N_VClone", 0)) { return (1); }

  res = N_VClone(uu);
  if (check_flag((void*)res, "N_VClone", 0)) { return (1); }

  constraints = N_VClone(uu);
  if (check_flag((void*)constraints, "N_VClone", 0)) { return (1); }

  data->pp = N_VClone(uu);
  if (check_flag((void*)data->pp, "N_VClone", 0)) { return (1); }

  id = N_VClone(uu);
  if (check_flag((void*)id, "N_VClone", 0)) { return (1); }

  /* Initialize uu, up. */

  SetInitialProfile(data, uu, up, id, res);

  /* Set constraints to all 1's for nonnegative solution values. */

  N_VConst(ONE, constraints);

  /* Assign various parameters. */

  t0   = ZERO;
  t1   = SUN_RCONST(0.01);
  rtol = ZERO;
  atol = SUN_RCONST(1.0e-8);

  /* Call IDACreate and IDAMalloc to initialize solution */

  mem = IDACreate(ctx);
  if (check_flag((void*)mem, "IDACreate", 0)) { return (1); }

  ier = IDASetUserData(mem, data);
  if (check_flag(&ier, "IDASetUserData", 1)) { return (1); }

  /* Set which components are algebraic or differential */
  ier = IDASetId(mem, id);
  if (check_flag(&ier, "IDASetId", 1)) { return (1); }

  ier = IDASetConstraints(mem, constraints);
  if (check_flag(&ier, "IDASetConstraints", 1)) { return (1); }
  N_VDestroy(constraints);

  ier = IDAInit(mem, resHeat, t0, uu, up);
  if (check_flag(&ier, "IDAInit", 1)) { return (1); }

  ier = IDASStolerances(mem, rtol, atol);
  if (check_flag(&ier, "IDASStolerances", 1)) { return (1); }

  // Initialize a ReSolve DEVICE memory space.
  ReSolve::memory::MemorySpace memspace = ReSolve::memory::DEVICE;

  /* Create ReSolve SUNMatrix for use in linear solves */
  nnz = MGRID * MGRID + ((MGRID-2)*(MGRID-2)*4);
  A   = SUNMatrix_ReSolve(NEQ, NEQ, nnz, memspace, ctx);
  if (check_flag((void*)A, "SUNMatrix_ReSolve", 0)) { return (1); }

  /* Set up the sparsity structure */
  jacHeatInit(A);

  /* Set up the ReSolve Linear Solver workspace */
  ReSolve::LinAlgWorkspaceCUDA workspace;
  std::string refactor = "cusolverrf";
  workspace.initializeHandles();

  /* ReSolve direct solver instatiation */
  ReSolve::SystemSolver solver(&workspace,
                               "klu",    // factorization
                               refactor, // refactorization
                               refactor, // triangular solve
                               "none",   // preconditioner (always 'none' here)
                               "none"); // iterative refinement
   
  /* Create ReSolve linear solver */
  LS = SUNLinSol_ReSolve(&solver, A, memspace, ctx);
  if (check_flag((void*)LS, "SUNLinSol_ReSolve", 0)) { return (1); }

  /* Attach the matrix and linear solver */
  ier = IDASetLinearSolver(mem, LS, A);
  if (check_flag(&ier, "IDASetLinearSolver", 1)) { return (1); }

  /* Set the user-supplied Jacobian routine */
  ier = IDASetJacFn(mem, jacHeat);
  if (check_flag(&ier, "IDASetJacFn", 1)) { return (1); }

  /* Call IDACalcIC to correct the initial values. */
  ier = IDACalcIC(mem, IDA_YA_YDP_INIT, t1);
  if (check_flag(&ier, "IDACalcIC", 1)) { return (1); }

  /* Print output heading. */
  PrintHeader(rtol, atol);

  PrintOutput(mem, t0, uu);

  /* Loop over output times, call IDASolve, and print results. */

  for (tout = t1, iout = 1; iout <= NOUT; iout++, tout *= TWO)
  {
    ier = IDASolve(mem, tout, &tret, uu, up, IDA_NORMAL);
    if (check_flag(&ier, "IDASolve", 1)) { return (1); }

    PrintOutput(mem, tret, uu);
  }

  /* Print remaining counters and free memory. */
  ier = IDAGetNumErrTestFails(mem, &netf);
  check_flag(&ier, "IDAGetNumErrTestFails", 1);
  ier = IDAGetNumNonlinSolvConvFails(mem, &ncfn);
  check_flag(&ier, "IDAGetNumNonlinSolvConvFails", 1);
  printf("\n netf = %ld,   ncfn = %ld \n", netf, ncfn);

  /* Free Memory */

  IDAFree(&mem);
  SUNLinSolFree(LS);

  N_VDestroy(uu);
  N_VDestroy(up);
  N_VDestroy(res);
  N_VDestroy(id);

  N_VDestroy(data->pp);
  free(data);

  SUNContext_Free(&ctx);

  return (0);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDA
 *--------------------------------------------------------------------
 */

/*
 * resHeat: heat equation system residual function (user-supplied)
 * This uses 5-point central differencing on the interior points, and
 * includes algebraic equations for the boundary values.
 * So for each interior point, the residual component has the form
 *    res_i = u'_i - (central difference)_i
 * while for each boundary point, it is res_i = u_i.
 */

int resHeat(sunrealtype tt, N_Vector uu, N_Vector up, N_Vector rr, void* user_data)
{
  sunindextype mm;
  sunrealtype coeff;
  UserData data;

  const sunrealtype* uu_data = N_VGetDeviceArrayPointer_Cuda(uu);
  const sunrealtype* up_data = N_VGetDeviceArrayPointer_Cuda(up);
  sunrealtype* rr_data       = N_VGetDeviceArrayPointer_Cuda(rr);

  data = (UserData)user_data;

  coeff = data->coeff;
  mm    = data->mm;

  unsigned block = 256;
  unsigned grid  = (mm * mm + block - 1) / block;

  resHeatKernel<<<grid, block>>>(uu_data, up_data, rr_data, mm, coeff);

  return (0);
}

// Set up the sparsity structure of the Jacobian matrix
int jacHeatInit(SUNMatrix JJ)
{
  sunrealtype dx   = ONE / (MGRID - ONE);
  sunrealtype beta = SUN_RCONST(4.0) / (dx * dx) + 1;
  sunindextype i, j, k, sum = 0, entry = 0;

  // Get pointers
  sunindextype* rowptrs = SUNMatrix_ReSolve_IndexPointers(JJ, ReSolve::memory::HOST);
  sunindextype* colvals = SUNMatrix_ReSolve_IndexValues(JJ, ReSolve::memory::HOST);
  sunrealtype*  data    = SUNMatrix_ReSolve_Data(JJ, ReSolve::memory::HOST);

  // Fill the matrix
  for (k = 0; k < NEQ; k++)
  {
    i = k/MGRID;
    j = k%MGRID;

    // Check if exterior point
    if (j == 0 || j == MGRID - 1 || i == 0 || i == MGRID - 1)
    {
      sum = sum + 1;
      rowptrs[k+1] = sum;
      colvals[entry] = k; 
      data[entry] = ONE;
      entry = entry + 1;
    }
    else
    {
      sum = sum + 5;
      rowptrs[k+1] = sum;
      colvals[entry] = k - MGRID;
      colvals[entry + 1] = k - 1;
      colvals[entry + 2] = k;
      colvals[entry + 3] = k + 1;
      colvals[entry + 4] = k + MGRID;
      data[entry] = -ONE/(dx*dx);
      data[entry + 1] = -ONE/(dx*dx);
      data[entry + 2] = beta;
      data[entry + 3] = -ONE/(dx*dx);
      data[entry + 4] = -ONE/(dx*dx);
      entry = entry + 5;
    }
  }
  rowptrs[NEQ+1] = entry;
  // Set to updated
  SUNMatrix_ReSolve_SetUpdated(JJ, ReSolve::memory::HOST);

  ReSolve::memory::MemorySpace memspace = SUNMatrix_ReSolve_MemorySpace(JJ);

  // Sync to device if necessary
  if (memspace != ReSolve::memory::HOST)
  {
    SUNMatrix_ReSolve_SyncData(JJ, memspace);
  }

  return (0);
}

/* Jacobian for MGRID>=4 */
int jacHeat(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp,
            N_Vector resvec, SUNMatrix JJ, void* user_data, N_Vector tempv1,
            N_Vector tempv2, N_Vector tempv3)
{
  sunindextype mm;
  UserData data;

  // Get device data pointer and index pointers for indexing
  sunrealtype* jac_data = SUNMatrix_ReSolve_Data(JJ, ReSolve::memory::DEVICE);
  sunindextype* rowptrs = SUNMatrix_ReSolve_IndexPointers(JJ, ReSolve::memory::DEVICE);

  data = (UserData)user_data;

  mm = data->mm;

  unsigned block = 256;
  unsigned grid  = (mm * mm + block - 1) / block;

  jacHeatKernel<<<grid, block>>>(jac_data, rowptrs, cj); 
  
  // Set to updated
  SUNMatrix_ReSolve_SetUpdated(JJ, ReSolve::memory::DEVICE);
  SUNMatrix_ReSolve_SyncData(JJ, ReSolve::memory::HOST);

  return (0);
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * SetInitialProfile: routine to initialize u and up vectors.
 */

static int SetInitialProfile(UserData data, N_Vector uu, N_Vector up, N_Vector id, N_Vector res)
{
  sunindextype mm, i, j;
  sunrealtype xfact, yfact, *udata, *updata, *iddata;

  mm = data->mm;

  udata = N_VGetHostArrayPointer_Cuda(uu);

  /* Initialize uu on all grid points. */
  for (j = 0; j < mm; j++)
  {
    yfact = data->dx * j;
    for (i = 0; i < mm; i++)
    {
      xfact             = data->dx * i;
      udata[mm * j + i] = SUN_RCONST(16.0) * xfact * (ONE - xfact) * yfact *
                          (ONE - yfact);
    }
  }

  N_VCopyToDevice_Cuda(uu);

  /* Initialize up vector to 0. */
  N_VConst(ZERO, up);

  /* resHeat sets res to negative of ODE RHS values at interior points. */
  resHeat(ZERO, uu, up, res, data);

  /* Copy -res into up to get correct interior initial up values. */
  N_VScale(-ONE, res, up);

  /* Set up at boundary points to zero. */
  updata = N_VGetDeviceArrayPointer_Cuda(up);
  iddata = N_VGetDeviceArrayPointer_Cuda(id);

  unsigned block = 256;
  unsigned grid  = (mm * mm + block - 1) / block;

  setInitHeatKernel<<<grid, block>>>(updata, iddata, mm);

  return (0);
}

/*
 * Print first lines of output (problem description)
 */

static void PrintHeader(sunrealtype rtol, sunrealtype atol)
{
  printf("\nidaHeat2D_resolve_cuda: Heat equation, serial example problem for IDA \n");
  printf("         Discretized heat equation on 2D unit square. \n");
  printf("         Zero boundary conditions,");
  printf(" polynomial initial conditions.\n");
  printf("         Mesh dimensions: %d x %d", MGRID, MGRID);
  printf("        Total system size: %d\n\n", NEQ);
#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("Tolerance parameters:  rtol = %Lg   atol = %Lg\n", rtol, atol);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("Tolerance parameters:  rtol = %g   atol = %g\n", rtol, atol);
#else
  printf("Tolerance parameters:  rtol = %g   atol = %g\n", rtol, atol);
#endif
  printf("Constraints set to force all solution components >= 0. \n");
  printf("Linear solver: ReSolve System Solver using KLU \n");

  /* Print output table heading and initial line of table. */
  printf("\n   Output Summary (umax = max-norm of solution) \n\n");
  printf("  time       umax     k  nst  nni  nje   nre     h       \n");
  printf(" .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . \n");
}

/*
 * PrintOutput: print max norm of solution and current solver statistics
 */

static void PrintOutput(void* mem, sunrealtype t, N_Vector uu)
{
  sunrealtype hused, umax;
  long int nst, nni, nje, nre;
  int kused, ier;

  umax = N_VMaxNorm(uu);

  ier = IDAGetLastOrder(mem, &kused);
  check_flag(&ier, "IDAGetLastOrder", 1);
  ier = IDAGetNumSteps(mem, &nst);
  check_flag(&ier, "IDAGetNumSteps", 1);
  ier = IDAGetNumNonlinSolvIters(mem, &nni);
  check_flag(&ier, "IDAGetNumNonlinSolvIters", 1);
  ier = IDAGetNumResEvals(mem, &nre);
  check_flag(&ier, "IDAGetNumResEvals", 1);
  ier = IDAGetLastStep(mem, &hused);
  check_flag(&ier, "IDAGetLastStep", 1);
  ier = IDAGetNumJacEvals(mem, &nje);
  check_flag(&ier, "IDAGetNumJtimesEvals", 1);

#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf(" %5.2Lf %13.5Le  %d  %3ld  %3ld  %3ld  %4ld  %9.2Le \n", t, umax,
         kused, nst, nni, nje, nre, hused);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf(" %5.2f %13.5e  %d  %3ld  %3ld  %3ld  %4ld  %9.2e \n", t, umax, kused,
         nst, nni, nje, nre, hused);
#else
  printf(" %5.2f %13.5e  %d  %3ld  %3ld  %3ld  %4ld  %9.2e \n", t, umax, kused,
         nst, nni, nje, nre, hused);
#endif
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

static int check_flag(void* flagvalue, const char* funcname, int opt)
{
  int* errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }
  else if (opt == 1)
  {
    /* Check if flag < 0 */
    errflag = (int*)flagvalue;
    if (*errflag < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return (1);
    }
  }
  else if (opt == 2 && flagvalue == NULL)
  {
    /* Check if function returned NULL pointer - no memory allocated */
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}
