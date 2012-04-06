#include <dslash_constants.h>

/*

// double precision
#define READ_SPINOR READ_SPINOR_DOUBLE
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define SPINORTEX spinorTexDouble
#endif
#define SPINOR_DOUBLE
template <typename FloatN>
static inline __device__ void packFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, const float *inNorm,
                                                 const int &idx, const int &face_idx, const int &face_volume, const int &face_num)

#include "smearing_core.h"

#undef READ_SPINOR
#undef SPINORTEX
#undef SPINOR_DOUBLE
#endif
 */
 
  static inline __device__ void spinorSmearWilsonKernel_dp8(double2 *out, double2* in1, double2* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_sp8(float4 *out, float4* in1, float4* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_hp8(short4 *out, short4* in1, short4* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_dp12(double2 *out, double2* in1, double2* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_sp12(float4 *out, float4* in1, float4* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_hp12(short4 *out, short4* in1, short4* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_dp18(double2 *out, double2* in1, double2* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_sp18(float4 *out, float4* in1, float4* in2, const float& alpha_local, const float& alpha_NN){}
static inline __device__ void spinorSmearWilsonKernel_hp18(short4 *out, short4* in1, short4* in2, const float& alpha_local, const float& alpha_NN){}


__global__ void spinorSmearWilson(cudaColorSpinorField &out, cudaColorSpinorField &in, QudaSpinorSmearParam *smear_param, const QudaReconstructType reconstruct) {

 void *outE = (void *)out.Even().V();
 void *outO = (void *)out.Odd().V();
 void *inE = (void *)in.Even().V();
 void *inO = (void *)in.Odd().V();

  const float alpha_local = smear_param->alpha_local;
  const float alpha_NN = smear_param->alpha_NN;
  const int nsteps = smear_param->nsteps;

  void *outputE, *outputO, *inputE, *inputO, *tmpE, *tmpO;
  outputE = outE;
  outputO = outO;
  inputE = inE;
  inputO = inO;

  //Smear over n steps
  for (int i = 0; i < nsteps; i++) {

    //Switch on the gauge field reconstruction and the precision
    //to decide which kernels to call.
    switch(reconstruct) {
    case QUDA_RECONSTRUCT_8:

      switch(in.Precision()) {

      case QUDA_DOUBLE_PRECISION:
	spinorSmearWilsonKernel_dp8((double2 *)outputE, (double2 *)inputE, (double2 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_dp8((double2 *)outputO, (double2 *)inputO, (double2 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_SINGLE_PRECISION:
	spinorSmearWilsonKernel_sp8((float4 *)outputE, (float4 *)inputE, (float4 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_sp8((float4 *)outputO, (float4 *)inputO, (float4 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_HALF_PRECISION:
	spinorSmearWilsonKernel_hp8((short4 *)outputE, (short4 *)inputE, (short4 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_hp8((short4 *)outputO, (short4 *)inputO, (short4 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_INVALID_PRECISION:
	errorQuda("Invalid precision in spinorSmearWilson.\n");
	break;
      }
      break;
    case QUDA_RECONSTRUCT_12:

      switch(in.Precision()) {

      case QUDA_DOUBLE_PRECISION:
	spinorSmearWilsonKernel_dp12((double2 *)outputE, (double2 *)inputE, (double2 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_dp12((double2 *)outputO, (double2 *)inputO, (double2 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_SINGLE_PRECISION:
	spinorSmearWilsonKernel_sp12((float4 *)outputE, (float4 *)inputE, (float4 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_sp12((float4 *)outputO, (float4 *)inputO, (float4 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_HALF_PRECISION:
	spinorSmearWilsonKernel_hp12((short4 *)outputE, (short4 *)inputE, (short4 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_hp12((short4 *)outputO, (short4 *)inputO, (short4 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_INVALID_PRECISION:
	errorQuda("Invalid precision in spinorSmearWilson.\n");
	break;
      }
      break;

    case QUDA_RECONSTRUCT_NO:

      switch(in.Precision()) {

      case QUDA_DOUBLE_PRECISION:
	spinorSmearWilsonKernel_dp18((double2 *)outputE, (double2 *)inputE, (double2 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_dp18((double2 *)outputO, (double2 *)inputO, (double2 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_SINGLE_PRECISION:
	spinorSmearWilsonKernel_sp18((float4 *)outputE, (float4 *)inputE, (float4 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_sp18((float4 *)outputO, (float4 *)inputO, (float4 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_HALF_PRECISION:
	spinorSmearWilsonKernel_hp18((short4 *)outputE, (short4 *)inputE, (short4 *)inputO, alpha_local, alpha_NN);
	spinorSmearWilsonKernel_hp18((short4 *)outputO, (short4 *)inputO, (short4 *)inputE, alpha_local, alpha_NN);
	break;

      case QUDA_INVALID_PRECISION:
	errorQuda("Invalid precision in spinorSmearWilson.\n");
	break;
      }
      break;

    default:
      errorQuda("Gauge field reconstruction type = %d not supported in smearing.\n", reconstruct);
      break;
    }

    //Swap the input and output pointers for the next step
    tmpE = inputE;
    tmpO = inputO;
    inputE = outputE;
    inputO = outputO;
    outputE = tmpE;
    outputO = tmpO;
  }

  //If odd number of smearing steps, then *inE and *inO
  //already contain desired results.
  //If even number of smearing steps, then we must
  //copy the data back to inE and inO from outE and outO
  if(nsteps%2 == 0) {
  }
  printf("reconstruct = %d, precision = %d\n", reconstruct, in.Precision());
  
}

void spinorSmear(cudaColorSpinorField &out, cudaColorSpinorField &in, QudaSpinorSmearParam *smear_param, cudaGaugeField &gauge) {
  if(!initDslash) initDslashConstants(gauge, in.Stride());
  if(in.Nspin() == 4) {
    spinorSmearWilson(out, in, smear_param, gauge.Reconstruct());
  }
  else {
    errorQuda("Staggered spinor smearing not implemented (yet).\n"); 
    //spinorSmearStaggered(out, in smear_param, reconstruct);
  }
}
