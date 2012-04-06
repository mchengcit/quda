#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <blas_reference.h>
#include <smearing_reference.h>
#include "misc.h"

#include "face_quda.h"

#ifdef QMP_COMMS
#include <qmp.h>
#endif

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// Wilson, clover-improved Wilson, and twisted mass are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;

extern char latfile[];

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d \n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     commDimPartitioned(0),
	     commDimPartitioned(1),
	     commDimPartitioned(2),
	     commDimPartitioned(3)); 
  
  return ;
  
}

extern void usage(char** );
int main(int argc, char **argv)
{
  /*
  int ndim=4, dims[4] = {1, 1, 1, 1};
  char dimchar[] = {'X', 'Y', 'Z', 'T'};
  char *gridsizeopt[] = {"--xgridsize", "--ygridsize", "--zgridsize", "--tgridsize"};

  for (int i=1; i<argc; i++) {
    for (int d=0; d<ndim; d++) {
      if (!strcmp(argv[i], gridsizeopt[d])) {
	if (i+1 >= argc) {
	  printf("Usage: %s <args>\n", argv[0]);
	  printf("%s\t Set %c comms grid size (default = 1)\n", gridsizeopt[d], dimchar[d]); 
	  exit(1);
	}     
	dims[d] = atoi(argv[i+1]);
	if (dims[d] <= 0 ) {
	  printf("Error: Invalid %c grid size\n", dimchar[d]);
	  exit(1);
	}
	i++;
	break;
      }
    }
  }
  */



  int i;
  for (i =1;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
    

  }


  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }


  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);

  // *** QUDA parameters begin here.


  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH) {
    printf("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = QUDA_HALF_PRECISION;

  double r = 1.0;
  int niter = 10;

  QudaSpinorSmearParam smear_param = newQudaSpinorSmearParam();
  
  smear_param.type = QUDA_JACOBI_SMEAR;
  smear_param.alpha_local = 1.0/(1+6*r);
  smear_param.alpha_NN = r/(1+6*r);
  smear_param.nsteps = niter;
  smear_param.cpu_prec = cpu_prec;
  smear_param.cuda_prec = cuda_prec;
  smear_param.nSpin = 4;
  smear_param.pad = 0;
  smear_param.diracOrder = QUDA_DIRAC_ORDER;
  smear_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  smear_param.verbosity = QUDA_VERBOSE;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
 
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = dslash_type;

  double mass = -0.2180;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.mu = 0.1;
    inv_param.twist_flavor = QUDA_TWIST_MINUS;
  }

  inv_param.solution_type = QUDA_MATPC_SOLUTION;
  inv_param.solve_type = QUDA_NORMEQ_PC_SOLVE;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
  inv_param.gcrNkrylov = 30;
  inv_param.tol = 5e-7;
  inv_param.maxiter = 30;
  inv_param.reliable_delta = 1e-1; // ignored by multi-shift solver

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = QUDA_INVALID_INVERTER;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  inv_param.prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.dirac_tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;
  inv_param.preserve_dirac = QUDA_PRESERVE_DIRAC_YES;

  gauge_param.ga_pad = 0; // 24*24*24/2;
  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.verbosity = QUDA_VERBOSE;

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  setDims(gauge_param.X);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field (type 1) or unite gauge (type 0)
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

#if 0  
  printGaugeElement(gauge[0], 0, gauge_param.cpu_prec);
  printGaugeElement(gauge[1], 0, gauge_param.cpu_prec);
  printGaugeElement(gauge[2], 0, gauge_param.cpu_prec);
  printGaugeElement(gauge[3], 0, gauge_param.cpu_prec);
#endif  



  void *spinorIn = malloc(V*spinorSiteSize*sSize);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize);

  void *spinorOut = NULL;
  spinorOut = malloc(V*spinorSiteSize*sSize);

  //type = 0, point source at (0,0,0,0) s = 0, c= 0
  //type = 1 Random color-spinor field
  construct_spinor_field(spinorIn, 1, inv_param.cpu_prec);

#if 0
  printf("Spinor at (0,0,0,0):\n");
  printSpinorElement(spinorIn, 0, inv_param.cpu_prec);

  for(int dir = 0; dir < 8; dir++) {
    int j;
    int nb = 1;
    switch (dir) {
    case 0: j = neighborIndexFullLattice(0, 0, 0, 0, +nb); break;
    case 1: j = neighborIndexFullLattice(0, 0, 0, 0, -nb); break;
    case 2: j = neighborIndexFullLattice(0, 0, 0, +nb, 0); break;
    case 3: j = neighborIndexFullLattice(0, 0, 0, -nb, 0); break;
    case 4: j = neighborIndexFullLattice(0, 0, +nb, 0, 0); break;
    case 5: j = neighborIndexFullLattice(0, 0, -nb, 0, 0); break;
    case 6: j = neighborIndexFullLattice(0, +nb, 0, 0, 0); break;
    case 7: j = neighborIndexFullLattice(0, -nb, 0, 0, 0); break;
    default: j = -1; break;
    }

    printf("dir = %d\n", dir);
    if (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) 
      printSpinorElement((double *)spinorIn, j, inv_param.cpu_prec);
    else
      printSpinorElement((float *)spinorIn, j, inv_param.cpu_prec);
  }
#endif


  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  //Call the Quda version
  smearSpinor(spinorOut, spinorIn, &smear_param);

  SmearJacobi(spinorOut, gauge, spinorIn, inv_param.cpu_prec, r, niter);

#if 0
  printf("Spinor at (0,0,0,0):\n");
  printSpinorElement(spinorOut, 0, inv_param.cpu_prec);

  for(int dir = 0; dir < 8; dir++) {
    int j;
    int nb = 1;
    switch (dir) {
    case 0: j = neighborIndexFullLattice(0, 0, 0, 0, +nb); break;
    case 1: j = neighborIndexFullLattice(0, 0, 0, 0, -nb); break;
    case 2: j = neighborIndexFullLattice(0, 0, 0, +nb, 0); break;
    case 3: j = neighborIndexFullLattice(0, 0, 0, -nb, 0); break;
    case 4: j = neighborIndexFullLattice(0, 0, +nb, 0, 0); break;
    case 5: j = neighborIndexFullLattice(0, 0, -nb, 0, 0); break;
    case 6: j = neighborIndexFullLattice(0, +nb, 0, 0, 0); break;
    case 7: j = neighborIndexFullLattice(0, -nb, 0, 0, 0); break;
    default: j = -1; break;
    }

    printf("dir = %d\n", dir);
    if (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) 
      printSpinorElement((double *)spinorOut, j, inv_param.cpu_prec);
    else
      printSpinorElement((float *)spinorOut, j, inv_param.cpu_prec);
  }
#endif

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printf("%d smearing steps in %g seconds\n", niter,time0); 
    
  freeGaugeQuda();

  // finalize the QUDA library
  endQuda();

  // end if the communications layer
  endCommsQuda();

  return 0;
}
