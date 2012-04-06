#ifndef _SMEARING_REFERENCE_H
#define _SMEARING_REFERENCE_H

#include <enum_quda.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern int Z[4];
  extern int Vh;
  extern int V;

  void setDims(int *);

  void construct_spinor_field(void *spinor, int type, QudaPrecision precision);
  void SmearJacobi(void *res, void **gaugeFull, void *spinorField, QudaPrecision precision, double r, int steps);

#ifdef __cplusplus
}
#endif

#endif // _SMEARING_REFERENCE_H
