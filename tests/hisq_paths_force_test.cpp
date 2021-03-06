#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include "test_util.h"
#include "gauge_field.h"
#include "fat_force_quda.h"
#include "misc.h"
#include "hisq_force_reference.h"
#include "hisq_force_quda.h"
#include "hisq_force_utils.h"
#include "hw_quda.h"
#include <sys/time.h>

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))



#include "fermion_force_reference.h"
using namespace hisq::fermion_force;

extern void usage(char** argv);
static int device = 0;
cudaGaugeField *cudaGauge = NULL;
cpuGaugeField  *cpuGauge  = NULL;

cudaGaugeField *cudaForce = NULL;
cpuGaugeField  *cpuForce = NULL;

cudaGaugeField *cudaMom = NULL;
cpuGaugeField *cpuMom  = NULL;
cpuGaugeField *refMom  = NULL;

static FullHw cudaHw;
static QudaGaugeParam gaugeParam;
static void* hw; // the array of half_wilson_vector


cpuGaugeField *cpuOprod = NULL;
cudaGaugeField *cudaOprod = NULL;
cpuGaugeField *cpuLongLinkOprod = NULL;
cudaGaugeField *cudaLongLinkOprod = NULL;

int verify_results = 0;
int ODD_BIT = 1;
extern int xdim, ydim, zdim, tdim;

extern QudaPrecision prec;
extern QudaReconstructType link_recon;
QudaPrecision link_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision hw_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cpu_hw_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision mom_prec = QUDA_DOUBLE_PRECISION;



static void setPrecision(QudaPrecision precision)
{
  link_prec = precision;
  hw_prec = precision;
  cpu_hw_prec = precision;
  mom_prec = precision;
  return;
}

int Z[4];
int V;
int Vh;


void
setDims(int *X){
  V = 1;
  for(int dir=0; dir<4; ++dir){
    V *= X[dir];
    Z[dir] = X[dir];
  }
  Vh = V/2;
  return;
}


void
total_staple_io_flops(QudaPrecision prec, QudaReconstructType recon, double* io, double* flops)
{
  //total IO counting for the middle/side/all link kernels
  //Explanation about these numbers can be founed in the corresnponding kernel functions in
  //the hisq kernel core file
  int linksize = prec*recon;
  int cmsize = prec*18;
  
  int matrix_mul_flops = 198;
  int matrix_add_flops = 18;

  int num_calls_middle_link[6] = {24, 24, 96, 96, 24, 24};
  int middle_link_data_io[6][2] = {
    {3,6},
    {3,4},
    {3,7},
    {3,5},
    {3,5},
    {3,2}
  };
  int middle_link_data_flops[6][2] = {
    {3,1},
    {2,0},
    {4,1},
    {3,0},
    {4,1},
    {2,0}
  };


  int num_calls_side_link[2]= {192, 48};
  int side_link_data_io[2][2] = {
    {1, 6},
    {0, 3}
  };
  int side_link_data_flops[2][2] = {
    {2, 2},
    {0, 1}
  };



  int num_calls_all_link[2] ={192, 192};
  int all_link_data_io[2][2] = {
    {3, 8},
    {3, 6}
  };
  int all_link_data_flops[2][2] = {
    {6, 3},
    {4, 2}
  };

  
  double total_io = 0;
  for(int i = 0;i < 6; i++){
    total_io += num_calls_middle_link[i]
      *(middle_link_data_io[i][0]*linksize + middle_link_data_io[i][1]*cmsize);
  }
  
  for(int i = 0;i < 2; i++){
    total_io += num_calls_side_link[i]
      *(side_link_data_io[i][0]*linksize + side_link_data_io[i][1]*cmsize);
  }
  for(int i = 0;i < 2; i++){
    total_io += num_calls_all_link[i]
      *(all_link_data_io[i][0]*linksize + all_link_data_io[i][1]*cmsize);
  }	
  total_io *= V;


  double total_flops = 0;
  for(int i = 0;i < 6; i++){
    total_flops += num_calls_middle_link[i]
      *(middle_link_data_flops[i][0]*matrix_mul_flops + middle_link_data_flops[i][1]*matrix_add_flops);
  }
  
  for(int i = 0;i < 2; i++){
    total_flops += num_calls_side_link[i]
      *(side_link_data_flops[i][0]*matrix_mul_flops + side_link_data_flops[i][1]*matrix_add_flops);
  }
  for(int i = 0;i < 2; i++){
    total_flops += num_calls_all_link[i]
      *(all_link_data_flops[i][0]*matrix_mul_flops + all_link_data_flops[i][1]*matrix_add_flops);
  }	
  total_flops *= V;

  *io=total_io;
  *flops = total_flops;

  printfQuda("flop/byte =%.1f\n", total_flops/total_io);
  return ;  
}


void initDslashConstants(const cudaGaugeField &gauge, const int sp_stride);


// allocate memory
// set the layout, etc.
static void
hisq_force_init()
{
  initQuda(device);

  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;

  setDims(gaugeParam.X);


  gaugeParam.cpu_prec = link_prec;
  gaugeParam.cuda_prec = link_prec;
  gaugeParam.reconstruct = link_recon;

  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

  GaugeFieldParam gParam(0, gaugeParam);
  gParam.create = QUDA_NULL_FIELD_CREATE;

  cpuGauge = new cpuGaugeField(gParam);

  // this is a hack to get the gauge field to appear as a void** rather than void*
  void* siteLink_2d[4];
  for(int i=0;i < 4;i++){
    siteLink_2d[i] = malloc(V*gaugeSiteSize* gaugeParam.cpu_prec);
    if(siteLink_2d[i] == NULL){
      errorQuda("malloc failed for siteLink_2d\n");
    }
  }
  
  // fills the gauge field with random numbers
  createSiteLinkCPU(siteLink_2d, gaugeParam.cpu_prec, 1);
  
  for(int dir = 0; dir < 4; dir++){
    for(int i = 0;i < V; i++){
      char* src = (char*)siteLink_2d[dir];
      char* dst = (char*)cpuGauge->Gauge_p();
      memcpy(dst + (4*i+dir)*gaugeSiteSize*link_prec, src + i*gaugeSiteSize*link_prec, gaugeSiteSize*link_prec);   
    }
  }

  gParam.precision = gaugeParam.cuda_prec;
  gParam.reconstruct = link_recon;
  cudaGauge = new cudaGaugeField(gParam);

  // create the force matrix
  // cannot reconstruct, since the force matrix is not in SU(3)
  gParam.precision = gaugeParam.cpu_prec;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cpuForce = new cpuGaugeField(gParam); 
  memset(cpuForce->Gauge_p(), 0, 4*cpuForce->Volume()*gaugeSiteSize*gaugeParam.cpu_prec);

  gParam.precision = gaugeParam.cuda_prec;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaForce = new cudaGaugeField(gParam); 
  cudaMemset((void**)(cudaForce->Gauge_p()), 0, cudaForce->Bytes());

  // create the momentum matrix
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.precision = gaugeParam.cpu_prec;
  cpuMom = new cpuGaugeField(gParam);
  refMom = new cpuGaugeField(gParam);

  createMomCPU(cpuMom->Gauge_p(), mom_prec);


  memset(cpuMom->Gauge_p(), 0, cpuMom->Bytes());
  memset(refMom->Gauge_p(), 0, refMom->Bytes());

  gParam.precision = gaugeParam.cuda_prec;
  cudaMom = new cudaGaugeField(gParam); // Are the elements initialised to zero? - No!

  hw = malloc(4*cpuGauge->Volume()*hwSiteSize*gaugeParam.cpu_prec);
  if (hw == NULL){
    fprintf(stderr, "ERROR: malloc failed for hw\n");
    exit(1);
  }

  createHwCPU(hw, hw_prec);
  cudaHw = createHwQuda(gaugeParam.X, hw_prec);



  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.precision = gaugeParam.cpu_prec;
  cpuOprod = new cpuGaugeField(gParam);
  computeLinkOrderedOuterProduct(hw, cpuOprod->Gauge_p(), hw_prec, 1);

  cpuLongLinkOprod = new cpuGaugeField(gParam);
  computeLinkOrderedOuterProduct(hw, cpuLongLinkOprod->Gauge_p(), hw_prec, 3);


  gParam.precision = hw_prec;
  cudaOprod = new cudaGaugeField(gParam);
  cudaLongLinkOprod = new cudaGaugeField(gParam);

  for(int i = 0;i < 4; i++){
    free(siteLink_2d[i]);
  }
  return;
}


static void 
hisq_force_end()
{
  delete cudaMom;
  delete cudaForce;
  delete cudaGauge;
  delete cudaOprod;
  delete cudaLongLinkOprod;
  freeHwQuda(cudaHw);

  delete cpuGauge;
  delete cpuForce;
  delete cpuMom;
  delete refMom;
  delete cpuOprod;  
  delete cpuLongLinkOprod;
  free(hw);

  endQuda();

  return;
}

static int 
hisq_force_test(void)
{
  hisq_force_init();

  initDslashConstants(*cudaGauge, cudaGauge->VolumeCB());
  hisqForceInitCuda(&gaugeParam);


   
  float weight = 1.0;
  float act_path_coeff[6];

  act_path_coeff[0] = 0.625000;
  act_path_coeff[1] = -0.058479;
  act_path_coeff[2] = -0.087719;
  act_path_coeff[3] = 0.030778;
  act_path_coeff[4] = -0.007200;
  act_path_coeff[5] = -0.123113;


  double d_weight = 1.0;
  double d_act_path_coeff[6];
  for(int i=0; i<6; ++i){
    d_act_path_coeff[i] = act_path_coeff[i];
  }





  // copy the momentum field to the GPU
  cudaMom->loadCPUField(*refMom, QUDA_CPU_FIELD_LOCATION);
  // copy the gauge field to the GPU
  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  // copy the outer product field to the GPU
  cudaOprod->loadCPUField(*cpuOprod, QUDA_CPU_FIELD_LOCATION);
  // load the three-link outer product to the GPU
  cudaLongLinkOprod->loadCPUField(*cpuLongLinkOprod, QUDA_CPU_FIELD_LOCATION);

  loadHwToGPU(cudaHw, hw, cpu_hw_prec);

  struct timeval ht0, ht1;
  gettimeofday(&ht0, NULL);
  if (verify_results){
    if(cpu_hw_prec == QUDA_SINGLE_PRECISION){
      const float eps = 0.5;
      fermion_force_reference(eps, weight, 0, act_path_coeff, hw, cpuGauge->Gauge_p(), refMom->Gauge_p());
    }else if(cpu_hw_prec == QUDA_DOUBLE_PRECISION){
      const double eps = 0.5;
      fermion_force_reference(eps, d_weight, 0, d_act_path_coeff, hw, cpuGauge->Gauge_p(), refMom->Gauge_p());
    }
  }
  gettimeofday(&ht1, NULL);

  struct timeval t0, t1, t2, t3, t4, t5;

  gettimeofday(&t0, NULL);
  hisqStaplesForceCuda(d_act_path_coeff, gaugeParam, *cudaOprod, *cudaGauge, cudaForce);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  checkCudaError();
 
  gettimeofday(&t2, NULL);
  hisqLongLinkForceCuda(d_act_path_coeff[1], gaugeParam, *cudaLongLinkOprod, *cudaGauge, cudaForce);
  cudaThreadSynchronize();
  gettimeofday(&t3, NULL);
  checkCudaError();
  gettimeofday(&t4, NULL);
  hisqCompleteForceCuda(gaugeParam, *cudaForce, *cudaGauge, cudaMom);
  cudaThreadSynchronize();
  checkCudaError();
  gettimeofday(&t5, NULL);



  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);

  int res;
  res = compare_floats(cpuMom->Gauge_p(), refMom->Gauge_p(), 4*cpuMom->Volume()*momSiteSize, 1e-5, gaugeParam.cpu_prec);

  int accuracy_level = strong_check_mom(cpuMom->Gauge_p(), refMom->Gauge_p(), 4*cpuMom->Volume(), gaugeParam.cpu_prec);
  printf("Test %s\n",(1 == res) ? "PASSED" : "FAILED");

  double total_io;
  double total_flops;
  total_staple_io_flops(link_prec, link_recon, &total_io, &total_flops);
  
  float perf_flops = total_flops / (TDIFF(t0, t1)) *1e-9;
  float perf = total_io / (TDIFF(t0, t1)) *1e-9;
  printf("Staples time: %.2f ms, perf =%.2f GFLOPS, achieved bandwidth= %.2f GB/s\n", TDIFF(t0,t1)*1000, perf_flops, perf);
  printf("Staples time : %g ms\t LongLink time : %g ms\t Completion time : %g ms\n", TDIFF(t0,t1)*1000, TDIFF(t2,t3)*1000, TDIFF(t4,t5)*1000);
  printf("Host time (half-wilson fermion force) : %g ms\n", TDIFF(ht0, ht1)*1000);

  hisq_force_end();

  return accuracy_level;
}


static void
display_test_info()
{
  printf("running the following fermion force computation test:\n");
    
  printf("link_precision           link_reconstruct           space_dim(x/y/z)         T_dimension\n");
  printf("%s                       %s                         %d/%d/%d                  %d \n", 
	 get_prec_str(link_prec),
	 get_recon_str(link_recon), 
	 xdim, ydim, zdim, tdim);
  return ;
    
}

void
usage_extra(char** argv )
{
  printf("Extra options: \n");
  printf("    --verify                                  # Verify the GPU results using CPU results\n");
  return ;
}
int 
main(int argc, char **argv) 
{
  int i;
  for (i =1;i < argc; i++){
	
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }    

    if( strcmp(argv[i], "--verify") == 0){
      verify_results=1;
      continue;	    
    }	
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

#ifdef MULTI_GPU
    initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
#endif

  setPrecision(prec);

  display_test_info();
    
  int accuracy_level = hisq_force_test();


#ifdef MULTI_GPU
  endCommsQuda();
#endif

  if(accuracy_level >=3 ){
    return EXIT_SUCCESS;
  }else{
    return -1;
  }
  
}


