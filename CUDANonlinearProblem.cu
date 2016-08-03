#include "CUDANonlinearProblem.hpp"
#include <cassert>

__constant__ float dev_pU[2];
const int nDim = 2;

/* Class to define CUDA problem */
CUDANonlinearProblem::CUDANonlinearProblem(const int noReal, const float *pu, const float *mpParameters) {

  N = noReal;

  lambda = mpParameters[0];
  sigma  = 0.01f;

  U = new float [nDim];
  F = new float [nDim];
  for (int i=0;i<nDim;i++) {
    U[i] = pu[i];
    F[i] = 0.0f;
  }

  curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed( gen, (unsigned long long) clock() );

  cudaMalloc( &dev_pPars, N*sizeof(float));
  cudaMalloc( &dev_pf, N*nDim*sizeof(float));
  cudaMemcpyToSymbol( dev_pU, U, nDim*sizeof(float), 0, cudaMemcpyHostToDevice);

  noThreads = 1024;
  noBlocks  = (N+noThreads-1)/noThreads;

}

CUDANonlinearProblem::~CUDANonlinearProblem() {

  cudaFree( dev_pPars);
  cudaFree( dev_pf);

  curandDestroyGenerator( gen);

  delete[] U;
  delete[] F;
}

void CUDANonlinearProblem::Compute_F() {

  curandGenerateNormal( gen, dev_pPars, N, lambda, sigma);

  local_f<<<noBlocks,noThreads>>>(dev_pPars,dev_pf,N);
  cudaSum( dev_pf, N, noThreads);
  cudaMemcpy( F, dev_pf, 2*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0;i<nDim;i++) {
    F[i] /= N;
  }
}

void CUDANonlinearProblem::SetNoThreads( int number) {
  assert(number>0);
  assert(number<1024);
  noThreads = number;
  noBlocks  = (N+noThreads-1)/noThreads;
}

void CUDANonlinearProblem::SetVariance( const float variance) {
  assert(variance>0);
  sigma = variance;
}

void CUDANonlinearProblem::SetNoRealisations( int noReal) {
  assert(noReal>0);
  N = noReal;
}

void CUDANonlinearProblem::ResetSeed() {
  curandSetPseudoRandomGeneratorSeed( gen, (unsigned long long) clock() );
}
