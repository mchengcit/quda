//
// Auto-tuned blas CUDA parameters, generated by blas_test
//

static int blas_threads[30][3] = {
  {  64,  512,   64},  // Kernel  0: copyCuda (high source precision)
  { 576,  544,  320},  // Kernel  1: copyCuda (low source precision)
  {  96,  128,  128},  // Kernel  2: axpbyCuda
  { 160,  128,  128},  // Kernel  3: xpyCuda
  { 160,  448,  128},  // Kernel  4: axpyCuda
  {  96,  128,  128},  // Kernel  5: xpayCuda
  { 160,  128,  128},  // Kernel  6: mxpyCuda
  { 192,  448,  640},  // Kernel  7: axCuda
  {  64,  128,   96},  // Kernel  8: caxpyCuda
  {  96,  128,   64},  // Kernel  9: caxpbyCuda
  {  96,   96,   96},  // Kernel 10: cxpaypbzCuda
  { 448,   64,   64},  // Kernel 11: axpyBzpcxCuda
  { 512,   64,   64},  // Kernel 12: axpyZpbxCuda
  {  64,   96,   64},  // Kernel 13: caxpbypzYmbwCuda
  { 128,  256,  256},  // Kernel 14: normCuda
  { 128,  128,  256},  // Kernel 15: reDotProductCuda
  { 256,  256,  512},  // Kernel 16: axpyNormCuda
  { 256,  256,  512},  // Kernel 17: xmyNormCuda
  { 128,  128,  512},  // Kernel 18: cDotProductCuda
  { 256,  256,  256},  // Kernel 19: xpaycDotzyCuda
  { 128,  128,  128},  // Kernel 20: cDotProductNormACuda
  { 128,  128,  128},  // Kernel 21: cDotProductNormBCuda
  { 256,  256,  256},  // Kernel 22: caxpbypzYmbwcDotProductWYNormYCuda
  { 128,  128,   64},  // Kernel 23: cabxpyAxCuda
  { 256,  256,  256},  // Kernel 24: caxpyNormCuda
  { 256,  256,  256},  // Kernel 25: caxpyXmazNormXCuda
  { 256,  512,  256},  // Kernel 26: cabxpyAxNormCuda
  {  64,  128,  256},  // Kernel 27: caxpbypzCuda
  {  64,  128,  128},  // Kernel 28: caxpbypczpwCuda
  { 256,  128,  256}   // Kernel 29: caxpyDotzyCuda
};

static int blas_blocks[30][3] = {
  { 4096,   512,  8192},  // Kernel  0: copyCuda (high source precision)
  { 8192,  1024, 65536},  // Kernel  1: copyCuda (low source precision)
  { 2048, 16384, 65536},  // Kernel  2: axpbyCuda
  { 1024, 16384, 65536},  // Kernel  3: xpyCuda
  { 1024,  4096, 32768},  // Kernel  4: axpyCuda
  { 2048, 16384, 32768},  // Kernel  5: xpayCuda
  { 1024, 16384, 32768},  // Kernel  6: mxpyCuda
  { 1024,  4096,  8192},  // Kernel  7: axCuda
  { 2048, 65536, 65536},  // Kernel  8: caxpyCuda
  { 2048, 32768, 32768},  // Kernel  9: caxpbyCuda
  { 2048, 32768, 65536},  // Kernel 10: cxpaypbzCuda
  {  512, 32768, 32768},  // Kernel 11: axpyBzpcxCuda
  {  512, 32768, 32768},  // Kernel 12: axpyZpbxCuda
  { 4096, 65536, 65536},  // Kernel 13: caxpbypzYmbwCuda
  {   64,   256,  1024},  // Kernel 14: normCuda
  {  256,  1024,  1024},  // Kernel 15: reDotProductCuda
  { 2048,    64,  4096},  // Kernel 16: axpyNormCuda
  {65536,    64,  4096},  // Kernel 17: xmyNormCuda
  {  128,   512,   128},  // Kernel 18: cDotProductCuda
  {  512,  1024,  2048},  // Kernel 19: xpaycDotzyCuda
  {  128,   512,   512},  // Kernel 20: cDotProductNormACuda
  {  128,   512,  1024},  // Kernel 21: cDotProductNormBCuda
  {  512,   512,  1024},  // Kernel 22: caxpbypzYmbwcDotProductWYNormYCuda
  { 2048, 32768, 32768},  // Kernel 23: cabxpyAxCuda
  {  512,  2048,  2048},  // Kernel 24: caxpyNormCuda
  { 2048,  2048,  4096},  // Kernel 25: caxpyXmazNormXCuda
  { 2048,  2048,  4096},  // Kernel 26: cabxpyAxNormCuda
  { 4096, 65536, 32768},  // Kernel 27: caxpbypzCuda
  { 4096, 65536, 32768},  // Kernel 28: caxpbypczpwCuda
  {  512,  1024,  1024}   // Kernel 29: caxpyDotzyCuda
};