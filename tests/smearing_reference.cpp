#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <util_quda.h>

#include <test_util.h>
#include <blas_reference.h>
#include <smearing_reference.h>

#include <gauge_field.h>
#include <color_spinor_field.h>


static int mySpinorSiteSize = 24;
#include <dslash_util.h>

template <typename Float>
void xeqy(Float *x, Float *y, int cnt) {
  for(int i = 0; i < cnt; i++) {
    x[i] = y[i];
  }
}

template <typename Float>
void xeqay(Float *x, double a, Float* y, int cnt) {
  for(int i = 0; i < cnt; i++) {
    x[i] = a*y[i];
  }
}

template <typename Float>
void xpeqay(Float *x, double a, Float* y, int cnt) {
  for(int i = 0; i < cnt; i++) {
    x[i] += a*y[i];
  }
}

template <typename Float> 
static void constructRandomSpinor(Float *spinor) {
  Float *spinorOdd, *spinorEven;
  spinorEven = spinor;
  spinorOdd = spinor+Vh*spinorSiteSize;


  for (int i = 0; i < Vh; i++) {
    for (int s = 0; s < 4; s++) { //spinor components
      for (int c = 0; c < 3; c++) { //colors
	spinorEven[i*spinorSiteSize + s*(3*2) + c*(2) + 0] = rand() / (Float)RAND_MAX;
	spinorEven[i*spinorSiteSize + s*(3*2) + c*(2) + 1] = rand() / (Float)RAND_MAX;
	spinorOdd[i*spinorSiteSize + s*(3*2) + c*(2) + 0] = rand() / (Float)RAND_MAX;
	spinorOdd[i*spinorSiteSize + s*(3*2) + c*(2) + 1] = rand() / (Float)RAND_MAX;
      }
    }
  }
}

//Point spinor at (0,0,0,0), s=0, c=0
template <typename Float>
static void constructPointSpinor(Float *spinor) {
  for (int i = 0; i < V*spinorSiteSize; i++) {
	spinor[i] = 0.0;
  }
  spinor[0] = 1.0;
}

void construct_spinor_field(void *spinor, int type, QudaPrecision precision) {
  if(type==0) {
   if(precision == QUDA_DOUBLE_PRECISION) constructPointSpinor((double *)spinor);
    else constructPointSpinor((float *)spinor);
  }
  else {
    if(precision == QUDA_DOUBLE_PRECISION) constructRandomSpinor((double *)spinor);
    else constructRandomSpinor((float *)spinor);
  }
}

template <typename Float>
Float *spinorNeighborFullLattice(int i, int dir, Float *spinorField, int neighbor_distance) 
{
  int j;
  int nb = neighbor_distance;
  switch (dir) {
  case 0: j = neighborIndexFullLattice(i, 0, 0, 0, +nb); break;
  case 1: j = neighborIndexFullLattice(i, 0, 0, 0, -nb); break;
  case 2: j = neighborIndexFullLattice(i, 0, 0, +nb, 0); break;
  case 3: j = neighborIndexFullLattice(i, 0, 0, -nb, 0); break;
  case 4: j = neighborIndexFullLattice(i, 0, +nb, 0, 0); break;
  case 5: j = neighborIndexFullLattice(i, 0, -nb, 0, 0); break;
  case 6: j = neighborIndexFullLattice(i, +nb, 0, 0, 0); break;
  case 7: j = neighborIndexFullLattice(i, -nb, 0, 0, 0); break;
  default: j = -1; break;
  }
    
  return &spinorField[j*(spinorSiteSize)];
}

template <typename sFloat, typename gFloat>
void SmearJacobireference(sFloat *res, gFloat **gaugeFull, sFloat *spinorField, double r, int steps) {
  sFloat *tmpSpinor = (sFloat *)malloc(V*spinorSiteSize*sizeof(sFloat));
  sFloat *spinorIn, *spinorOut, *tmp;

  //Copy the contents of original spinor into tmpSpinor
  xeqy(tmpSpinor, spinorField, V*spinorSiteSize);
  spinorIn = tmpSpinor;
  spinorOut = res;

  gFloat *gaugeEven[4], *gaugeOdd[4];
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir]  = gaugeFull[dir]+Vh*gaugeSiteSize;
  }

  for(int iter = 0; iter < steps; iter++) { 

    xeqay(spinorOut, 1/(1+6*r), spinorIn, V*spinorSiteSize);

    for (int oddBit = 0; oddBit < 2; oddBit++) {
      for (int i = 0; i < Vh; i++) {
	int fullindex = oddBit*Vh + i;
	
	//Spatial smearing only
	for (int dir = 0; dir < 6; dir++) {
	  gFloat *gauge = gaugeLink(i, dir, oddBit, gaugeEven, gaugeOdd, 1);
	  sFloat *spinor = spinorNeighborFullLattice(fullindex, dir, spinorIn, 1);
	  sFloat gaugedSpinor[4*3*2];
	  
	  for (int s = 0; s < 4; s++) {
	    if (dir % 2 == 0) su3Mul(&gaugedSpinor[s*(3*2)], gauge, &spinor[s*(3*2)]);
	    else su3Tmul(&gaugedSpinor[s*(3*2)], gauge, &spinor[s*(3*2)]);
	  }
	
	  //Accumulate result from gaugedSpinor into spinorOut
	  xpeqay(&spinorOut[fullindex*(4*3*2)], r/(1+6*r), gaugedSpinor, 4*3*2);
	}
      }
    }

    //Swap the pointers
    tmp = spinorIn;
    spinorIn = spinorOut;
    spinorOut = tmp;
  }

  //Copy the contents of spinorIn into res, unless res already points to
  //spinorIn
  if(spinorIn != res) {
    xeqy(res, spinorIn, V*spinorSiteSize);
  }
  free(tmpSpinor);
}

void SmearJacobi(void * res, void ** gaugeFull, void * spinorField, QudaPrecision precision, double r, int steps) {
        if(precision == QUDA_DOUBLE_PRECISION) {
                SmearJacobireference((double *) res, (double **)gaugeFull, (double *)spinorField, r, steps);
        }
        else {
                SmearJacobireference((float *) res, (float **)gaugeFull, (float *)spinorField, r, steps);
        }
}

