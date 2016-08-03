#include <cmath>

#include "Fold.hpp"
#include <ctime>
#include <boost/scoped_ptr.hpp>

// Specialised constructor
Fold::Fold(const arma::vec* pParameters)
{
  // Read parameters
  mpParameters = new arma::vec(*pParameters);
  boost::scoped_ptr<int> p{new int{1}};
  p.reset();


  // CUDA stuff
  N = 100000000;

  // random number stuff
  sigma = 0.01f;
  curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed( gen, (unsigned long long) clock() );

  // set number of blocks and threads
  noThreads = 1024;
  noBlocks  = (N+noThreads-1)/noThreads;

  // allocate arrays
  cudaMalloc( &dev_p_pars, N*sizeof(float));
  cudaMalloc( &dev_p_f, N*nDim*sizeof(float));
  cudaMalloc( &dev_p_u, nDim*sizeof(float));
}

Fold::~Fold()
{
  delete mpParameters;

  // free arrays
  cudaFree( dev_p_pars);
  cudaFree( dev_p_f);
  cudaFree( dev_p_u);

  // clean up random stuff
  curandDestroyGenerator( gen);
}

void Fold::ComputeF(const arma::vec& pu, arma::vec& pf)
{
  const unsigned int nDim = pu.n_rows;
  float U[nDim];
  float lambda = (*mpParameters)(0);
  for (int i=0;i<nDim;i++) {
    U[i] = pu(i);
  };

  curandGenerateNormal( gen, dev_p_pars, N, lambda, sigma);
  cudaMemcpy( dev_p_u, U, nDim*sizeof(float), cudaMemcpyHostToDevice);

  local_f<<<noBlocks,noThreads>>>( dev_p_pars, dev_p_u, dev_p_f, N, nDim);
  cudaSum( dev_p_f, N, nDim);
  cudaMemcpy( U, dev_p_f, nDim*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i=0;i<nDim;i++) {
    U[i] /= N;
    pf(i) = U[i];
  }
}

void Fold::ComputeDFDU(const arma::vec& u, arma::mat& dfdu)
{
  dfdu(0,0) = -2.0*u(0);
  dfdu(0,1) = 0.0;
  dfdu(1,0) = 0.0;
  dfdu(1,1) = -1.0;
}

void Fold::SetNoThreads( int number) {
  assert(number>0);
  assert(number<1024);
  noThreads = number;
  noBlocks  = (N+noThreads-1)/noThreads;
}

void Fold::SetVariance( const float variance) {
  assert(variance>0);
  sigma = variance;
}

void Fold::SetNoRealisations( int noReal) {
  assert(noReal>0);
  N = noReal;
}

void Fold::ResetSeed() {
  curandSetPseudoRandomGeneratorSeed( gen, (unsigned long long) clock() );
}

__device__ float warpReduceSum( float val) {
  for (int offset = warpSize/2; offset>0; offset/=2) {
    val += __shfl_down( val, offset);
  }
  return val;
}

__device__ float blockReduceSum( float val) {

  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);  // Each warp performs partial reduction

  if (lane==0) {
    shared[wid] = val; // Write reduced value to shared memory
  }

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

  if (wid==0) {
    val = warpReduceSum(val); //Final reduce within first warp
  }

  return val;
}

__global__ void deviceReduceKernel(float *in, float *out, int N, const unsigned int nDim) {
  float sum;
  int i;
  # pragma unroll
  for (int j=0;j<nDim;j++) {
    sum = 0.0f;
    //reduce multiple elements per thread
    for (i = blockIdx.x*blockDim.x+threadIdx.x;
         i < N;
         i += blockDim.x*gridDim.x) {
      sum += in[i+j*N];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0) {
      out[blockIdx.x+j*gridDim.x] = sum;
    }
  }
}

__global__ void deviceReduceAltKernel(float *in, float *out, int N, const unsigned int nDim, const unsigned int offset) {
  float sum = 0.0f;
  unsigned int globalIndex = threadIdx.x+blockIdx.x*blockDim.x;
  for (int i=1;i<=nDim;i++) {
    if ((blockIdx.x<i*gridDim.x/nDim) && (blockIdx.x>=(i-1)*gridDim.x/nDim)) {
      if (globalIndex<i*N) {
        sum += in[globalIndex];
      }
    }
    globalIndex -= offset;
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0) {
    out[blockIdx.x] = sum;
  }
}

void cudaSumAlt( float *val, int N, const unsigned int nDim) {
  unsigned int noThreads = 1024;
  unsigned int noBlocks;
  unsigned int offset;
  float *valTemp;
  cudaMalloc( &valTemp, ((N+noThreads-1)/noThreads)*nDim*sizeof(float));
  unsigned int i = 0;
  while (N>1) {
    noBlocks = (N+noThreads-1)/noThreads;
    offset   = noBlocks*noThreads-N;
    noBlocks *= nDim;
    if (i%2==0) {
      deviceReduceAltKernel<<<noBlocks,noThreads>>>( val, valTemp, N, nDim, offset);
    }
    else {
      deviceReduceAltKernel<<<noBlocks,noThreads>>>( valTemp, val, N, nDim, offset);
    }
    N = noBlocks/nDim;
    i++;
  }
  if (i%2==1) {
    cudaMemcpy( val, valTemp, nDim*sizeof(float), cudaMemcpyDeviceToDevice);
  }
  cudaFree( valTemp);
}

void cudaSum( float *val, int N, const unsigned int nDim) {
  unsigned int noThreads = 1024;
  unsigned int noBlocks;
  float *valTemp;
  cudaMalloc( &valTemp, ((N+noThreads-1)/noThreads)*nDim*sizeof(float));
  unsigned int i = 0;
  while (N>1) {
    noBlocks = (N+noThreads-1)/noThreads;
    if (i%2==0) {
      deviceReduceKernel<<<noBlocks,noThreads>>>( val, valTemp, N, nDim);
    }
    else {
      deviceReduceKernel<<<noBlocks,noThreads>>>( valTemp, val, N, nDim);
    }
    N = noBlocks;
    i++;
  }
  if (i%2==1) {
    cudaMemcpy( val, valTemp, nDim*sizeof(float), cudaMemcpyDeviceToDevice);
  }
  cudaFree( valTemp);
}

__global__ void local_f( const float *dev_pPars, const float *dev_pU, float *dev_pf, int N, unsigned int nDim) {
  unsigned int globalIndex = threadIdx.x+blockIdx.x*blockDim.x;
  if (globalIndex<N) {
    float lambda = dev_pPars[globalIndex];
    dev_pf[globalIndex]   = lambda-pow(dev_pU[0],2);
    dev_pf[N+globalIndex] = -dev_pU[1];
  }
}
