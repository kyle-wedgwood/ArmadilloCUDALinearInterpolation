#include <cmath>
#include <ctime>
#include <armadillo>
#include <curand.h>
#include "eventDrivenMap.hpp"
#include "parameters.hpp"

// Specialised constructor
eventDrivenMap::eventDrivenMap(const arma::vec* pParameters)
{
  mpParameters = new arma::vec(*pParameters);

  // CUDA stuff
  mNoReal    = noReal;
  mNoThreads = noThreads;
  mNoBlocks  = (mNoReal+mNoThreads-1)/mNoThreads;
  mNoSpikes  = noSpikes;

  // allocate memory on CPU
  mpHost_v = (float*) malloc( mNoThreads*sizeof(float) );
  mpHost_s = (float*) malloc( mNoThreads*sizeof(float) );

  // allocate memory on GPU
  cudaMalloc( &mpDev_p, (*mpParameters).n_elem*sizeof(float) );
  cudaMalloc( &mpDev_v, mNoReal*mNoThreads*sizeof(float) );
  cudaMalloc( &mpDev_s, mNoReal*mNoThreads*sizeof(float) );
  cudaMalloc( &mpDev_w, mNoThreads*sizeof(float) );
  cudaMalloc( &mpDev_U, mNoReal*noSpikes*sizeof(float) );
  cudaMalloc( &mpDev_Z, noSpikes*sizeof(float) );
  cudaMalloc( &mpDev_firingTime, mNoReal*noSpikes*sizeof(float) );
  cudaMalloc( &mpDev_firingInd, mNoReal*noSpikes*sizeof(unsigned short) );
  cudaMalloc( &mpDev_spikeInd, mNoReal*noSpikes*sizeof(unsigned short) );
  cudaMalloc( &mpDev_spikeCount, mNoReal*sizeof(unsigned short) );
  cudaMalloc( &mpDev_lastSpikeInd, noSpikes*sizeof(unsigned short) );
  cudaMalloc( &mpDev_averages, 5*noSpikes*mNoReal*sizeof(float) );

  // Set up coupling kernel
  BuildCouplingKernel();

  // Copy parameters over
  cudaMemcpy(mpDev_p,mpParameters->begin(),(*mpParameters).n_elem*sizeof(float),cudaMemcpyHostToDevice);

  // initialise random number generators
  curandCreateGenerator( &mGen, CURAND_RNG_PSEUDO_DEFAULT);
  ResetSeed();
}

void eventDrivenMap::BuildCouplingKernel()
{
  float *w;
  w = (float*) malloc( mNoThreads*sizeof(float));
  for (int i=0;i<mNoThreads;++i)
  {
    float x = -L + (float)(2*L/mNoThreads)*i;
    w[i] = (a1*exp(-b1*abs(x))-a2*exp(-b2*abs(x)))*2*L/mNoThreads;
  }
  circshift(w,mNoThreads/2);
  cudaMemcpy(mpDev_w,w,mNoThreads*sizeof(float),cudaMemcpyHostToDevice);
  for (int i=0;i<mNoThreads;++i)
  {
    printf("%f\n",w[i]);
  }
  free(w);
}

eventDrivenMap::~eventDrivenMap()
{
  delete mpParameters;

  cudaFree(mpDev_p);
  cudaFree(mpDev_v);
  cudaFree(mpDev_s);
  cudaFree(mpDev_w);
  cudaFree(mpDev_U);
  cudaFree(mpDev_Z);
  cudaFree(mpDev_firingTime);
  cudaFree(mpDev_firingInd);
  cudaFree(mpDev_spikeInd);
  cudaFree(mpDev_spikeCount);
  cudaFree(mpDev_lastSpikeInd);
  cudaFree(mpDev_averages);

  curandDestroyGenerator(mGen);
}

void eventDrivenMap::ComputeF(const arma::vec& u, arma::vec& f)
{

  // Assuming that weight kernel does not change
  cudaMemcpy(mpDev_U,u.begin(),mNoSpikes*sizeof(float),cudaMemcpyHostToDevice);

  // Lift
  LiftKernel<<<mNoReal,mNoThreads>>>(mpDev_s,mpDev_v,mpDev_p,mpDev_U,mNoThreads,mNoReal,mNoSpikes);

  /*
  // Copy data to GPU
  //cudaMemcpy( dev_w, w, mNoThreads*sizeof(float), cudaMemcpyHostToDevice );
  // need to think about what to do with this bit
  //cudaMemcpy( dev_lastSpikeInd, lastSpikeInd, noSpikes*sizeof(unsigned short), cudaMemcpyHostToDevice );
  cudaMemset( dev_firingTime, 0.0f, mNoReal*M*sizeof(float) );
  cudaMemset( dev_firingInd, 0, mNoReal*M*sizeof(unsigned short) );
  cudaMemset( dev_spikeInd, 0, mNoReal*M*sizeof(unsigned short) );
  cudaMemset( dev_spikeCount, 0, mNoReal*sizeof(unsigned short) );

  // Evolve
  EvolveKernel<<<mNoReal,mNoThreads>>>(mpDev_v,mpDev_s,mpDev_w,mFinalTime,mpDev_firingInd,mpDev_firingTime,
      mpDev_spikeInd,mpDev_spikeCount,mpDev_lastSpikeInd,mNoThreads,mNoReal,nNoSpikes);

  // Restrict
  averagesSimultaneousBlocksKernel<<<mNoSpikes*mNoReal,mNoThreads>>>( mpDev_averages, mpDev_firingInd,
      mpDev_firingTime, mpDev_spikeInd, mpDev_spikeCount); // working
  Restrict<<<mNoReal,1>>>( dev_U, averages);
  realisationReductionKernelBlocks<<<noSpikes,N>>>( dev_V, dev_Z1); // working

  // Copy data back to CPU
  cudaMemcpy( Z, dev_Z, noSpikes*sizeof(float), cudaMemcpyDeviceToHost );

  // Compute F
  for (int i=0;i<noSpikes;++i)
  {
    f[i] = u[i]-Z[i];
  }
  */
}

void eventDrivenMap::SetTimeHorizon( const float T)
{
  assert(T>0);
  mFinalTime = T;
}

void eventDrivenMap::SetNoRealisations( int noReal)
{
  assert(noReal>0);
  mNoReal = noReal;
}

void eventDrivenMap::ResetSeed()
{
  curandSetPseudoRandomGeneratorSeed( mGen, (unsigned long long) clock() );
}


void eventDrivenMap::ZtoU( const float *Z, float *U) {
  U[0] = Z[0];
  U[1] = 0.0f;
  for (int i=2;i<=mNoSpikes;i++) {
    U[i] = Z[i-1];
  }
}

void eventDrivenMap::UtoZ( const float *U, float *Z) {
  Z[0] = U[0];
  for (int i=1;i<mNoSpikes;i++) {
    Z[i] = U[i+1];
  }
}

__global__ void LiftKernel( float *S, float *v, const float *par, const float *U, const
    unsigned int noThreads, const unsigned int noReal, const unsigned int noSpikes)
{
  int k = threadIdx.x + blockIdx.x*blockDim.x;
  int m;
  if(k<noThreads*noReal){

    //Define x-array
    float x = L - (float)(2*L/noThreads)*threadIdx.x;
    float s = 0.0f;
    float c = U[0];
    float beta = par[0];
    float dummyV, dummyS = 0.0f;

    // Lift Voltage
    # pragma unroll
    for(m=1; m<=noSpikes;m++){
      dummyV = ((x-c*U[m]>0.0f)*(((a1*beta*c)/((beta+c*b1)*(1.0f+c*b1)))* exp(c*U[m]*((1.0f+c*b1)/c))*exp(-b1*c*U[m])
              - ((a2*beta*c)/((beta+c*b2)*(1.0f+c*b2)))* exp(c*U[m]*((1.0f+c*b2)/c))*exp(-b2*c*U[m])+(a1*beta*c/(1.0f-beta))*exp(beta*U[m])*(1.0f/(beta+c*b1)+ 1.0f/(c*b1 - beta))*(exp((x/c)*(1.0f-beta))-exp(((c*U[m])/c)*(1.0f-beta)))-(a1*beta*c/((-beta+c*b1)*(1.0f-c*b1)))*exp(b1*c*U[m])*(exp(x*((1.0f-c*b1)/c))-exp(c*U[m]*((1.0f-c*b1)/c)))
               -(a2*beta*c/(1.0f-beta))*exp(beta*U[m])*(1.0f/(beta+c*b2) + 1.0f/(c*b2 - beta))*(exp((x/c)*(1.0f-beta))-exp((U[m])*(1.0f-beta)))
              +(a2*beta*c/((-beta+c*b2)*(1.0f-c*b2)))*exp(b2*c*U[m])*(exp(x*((1.0f-c*b2)/c))-exp(c*U[m]*((1.0f-c*b2)/c))))
               +
            (x-c*U[m]<=0.0f)*(((a1*beta*c)/((beta +c*b1)*(1.0f+c*b1)))*(exp(x*((1.0f+c*b1)/c)))*exp(-b1*c*U[m])
               - ((a2*beta*c)/((beta +c*b2)*(1.0f+c*b2)))*(exp(x*((1.0f+c*b2)/c)))*exp(-b2*c*U[m])))*exp(-x/c);

      s += dummyV - ((x - c*U[m])>0.0f)*exp(-(x-c*U[m])/c) + ((x-c*U[m])<=0.0f)*0.0f;

      dummyS += ((c*U[m]-x)>0.0f)*(beta*a1*(c/(beta +c*b1))*exp(b1*(x- c*U[m])) - beta*a2*(c/(beta+c*b2))*exp(b2*(x- c*U[m])))
        +((c*U[m]-x)<= 0.0f)*((2.0f*a1/b1)*(beta/(1.0f - ((beta*beta)/(c*c*b1*b1))))*exp(-(beta/c)*(x-c*U[m])) -beta*a1*(c/(-beta +c*b1))*(exp(b1*(c*U[m] - x)))
        - (2.0f*a2/b2)*(beta/(1.0f - ((beta*beta)/(c*c*b2*b2))))*exp(-(beta/c)*(x-c*U[m])) + beta*a2*(c/(-beta +c*b2))*(exp(b2*(c*U[m] - x))));
    }

    v[k] = I + s;
    v[k] *= (v[k]<1.0f);
    S[k] = dummyS;
  }

}

__device__ float fun( float t, float v, float s, float beta) {
  return v*exp(-t)+I*(1.0f-exp(-t))+s*exp(-t)/(1.0f-beta)*(exp((1.0f-beta)*t)-1.0f)-vth;
}

__device__ float dfun( float t, float v, float s, float beta) {
  return I*exp(-t)-v*exp(-t)+s*exp(-t)*exp(-t*(beta-1))+(s*exp(-t)*(exp(-t*(beta-1))-1.0f))/(beta-1);
}

__device__ float eventTime( float v0, float s0, float beta) {
  int decision;
  float f, df, estimatedTime = 0.0f;
  decision = (int) (v0>vth*pow(s0/(vth-I),1.0f/beta)+I*(1.0f-pow(s0/(vth-I),1.0f/beta))-(vth-I)/(beta-1.0f)*(s0/(vth-I)-pow(s0/(vth-I),1.0f/beta)));

  f  = fun( estimatedTime, v0, s0, beta)*decision;
  df = dfun( estimatedTime, v0, s0, beta);

  while (abs(f)>tol) {
    estimatedTime -= f/df;
    f  = fun( estimatedTime, v0, s0, beta);
    df = dfun( estimatedTime, v0, s0, beta);
  }

  return estimatedTime+100.0f*(1.0f-decision);

}
__global__ void EvolveKernel( float *v, float *s, const float *beta, const float *w, const float
    finalTime, unsigned short *firingInd, float *firingTime, unsigned short
    *spikeInd, unsigned short *spikeCount, unsigned short *lastSpikeInd)
{
  __shared__ float spikeTime[noThreads];
  __shared__ unsigned short index[noThreads];
  __shared__ unsigned short local_lastSpikeInd[noSpikes];
  unsigned int m = 0;
  float currentTime = 0.0f;
  unsigned int thread2, halfpoint, nTotalThreads;
  float local_v, local_s;
  unsigned short minIndex;
  float temp;

  // load values from global memory
  local_v = v[threadIdx.x+blockIdx.x*blockDim.x];
  local_s = s[threadIdx.x+blockIdx.x*blockDim.x];
  local_beta = beta[threadIdx.x+blockIdx.x*blockDim.x];

  if (threadIdx.x<noSpikes) {
    local_lastSpikeInd[threadIdx.x] = lastSpikeInd[threadIdx.x];
  }
  while ((currentTime<finalTime)&&(m<M)) {
    // find next firing times
    spikeTime[threadIdx.x] = eventTime(local_v,local_s,local_beta);
    index[threadIdx.x] = threadIdx.x;
    __syncthreads();

    // perform reduction to find minimum spike time
    nTotalThreads = blockDim.x;
    while (nTotalThreads>1) {
      halfpoint = (nTotalThreads>>1);
      if (threadIdx.x<halfpoint) {
        thread2 = threadIdx.x + halfpoint;

        temp = spikeTime[thread2];
        if (temp<spikeTime[threadIdx.x]) {
          spikeTime[threadIdx.x] = temp;
          index[threadIdx.x] = index[thread2];
        }
      }
      __syncthreads();
      nTotalThreads = halfpoint;
    }

    // update values to spike time
    local_v *= exp(-spikeTime[0]);
    local_v +=
      I*(1.0f-exp(-spikeTime[0]))+local_s*exp(-spikeTime[0])/(1.0f-local_beta)*(exp((1.0f-local_beta)*spikeTime[0])-1.0f);
    local_v *= (threadIdx.x!=index[0]);
    local_s *= exp(-local_beta*spikeTime[0]);
    local_s += local_beta*w[(threadIdx.x-index[0])*(threadIdx.x>=index[0])+(index[0]-threadIdx.x)*(threadIdx.x<index[0])];

    // store values
    minIndex = 0;
    if (threadIdx.x==0) {
      for (int i=1;i<noSpikes;i++) {
        minIndex += ((abs(index[0]-local_lastSpikeInd[i]))<(abs(index[0]-local_lastSpikeInd[minIndex])));
      }
      spikeInd[M*blockIdx.x+m]   = minIndex;
      firingInd[M*blockIdx.x+m]  = index[0];
      firingTime[M*blockIdx.x+m] = currentTime+spikeTime[0];
      local_lastSpikeInd[minIndex] = index[0];
    }
    currentTime += spikeTime[0];
    m++;
  }

  if (threadIdx.x==0) {
    spikeCount[blockIdx.x] = m;
  }
}

__global__ void Restrict( float *U, const float *averages) {
  __shared__ unsigned short noCross[noSpikes];
  __shared__ float xBar[noSpikes], tBar[noSpikes], tBarSq[noSpikes], xtBar[noSpikes];
  unsigned int i;
  float offset[noSpikes], speedNum, speedDenom, speed;

  for (i=0;i<noSpikes;i++) {
    noCross[i] = averages[5*noSpikes*blockIdx.x+4*noSpikes+i];
    xBar[i]    = averages[5*noSpikes*blockIdx.x+i]/noCross[i];
    tBar[i]    = averages[5*noSpikes*blockIdx.x+noSpikes+i]/noCross[i];
    tBarSq[i]  = averages[5*noSpikes*blockIdx.x+2*noSpikes+i]/noCross[i];
    xtBar[i]   = averages[5*noSpikes*blockIdx.x+3*noSpikes+i]/noCross[i];
    speedNum   += noCross[i]*(xtBar[i]-xBar[i]*tBar[i]);
    speedDenom += noCross[i]*(tBarSq[i]-tBar[i]*tBar[i]);
  }
  speed = speedNum/speedDenom*(2.0f*L/noThreads);

  # pragma unroll
  for (i=0;i<noSpikes;i++) {
    offset[i] = xBar[i]*2.0f*L/noThreads-speed*tBar[i];
  }
  U[noSpikes*blockIdx.x] = speed;

  # pragma unroll
  for (i=1;i<noSpikes;i++) {
    U[noSpikes*blockIdx.x+i] = (offset[0]-offset[i])/speed;
  }

}

__global__ void averagesSimultaneousBlocksKernel( float *averages, const unsigned int *firingInd, const float *firingTime, const unsigned int *spikeInd, const unsigned int *spikeCount) {
  unsigned int i, spikeNo = blockIdx.x % noSpikes, realNo = blockIdx.x/noSpikes;
  unsigned int index;
  unsigned int noLoad = (spikeCount[realNo]+blockDim.x-1)/blockDim.x;
  unsigned int count  = spikeCount[realNo];
  struct averaging val = { 0.0, 0, 0.0, 0.0, 0};

  for (i=0;i<noLoad;i++) {
    index = threadIdx.x+i*blockDim.x+noSpikes*realNo;
    val.t     += (threadIdx.x+i*blockDim.x < count) ? firingTime[index]*(spikeInd[index]==spikeNo) : 0.0f ;
    val.x     += (threadIdx.x+i*blockDim.x < count) ? firingInd[index]*(spikeInd[index]==spikeNo) : 0;
    val.count += (threadIdx.x+i*blockDim.x < count) ? spikeInd[index]==spikeNo : 0;
    val.tSq += val.t*val.t;
    val.xt  += val.t*val.x;
  }

  val = blockReduceSumSimultaneous( val);

  if (threadIdx.x==0) {
    averages[0*noSpikes*noReal+spikeNo*noReal+realNo] = (float) val.x/count;
    averages[1*noSpikes*noReal+spikeNo*noReal+realNo] = val.t/count;
    averages[2*noSpikes*noReal+spikeNo*noReal+realNo] = val.tSq/count;
    averages[3*noSpikes*noReal+spikeNo*noReal+realNo] = val.xt/count;
    averages[4*noSpikes*noReal+spikeNo*noReal+realNo] = (float) val.count/count;
  }
}

__global__ void realisationReductionKernelBlocks( float *U) {
  unsigned int i, spikeNo = blockIdx.x;
  unsigned int index;
  unsigned int noLoad = (noReal+blockDim.x-1)/blockDim.x;
  float average = 0.0f;

  for (i=0;i<noLoad;i++) {
    index = threadIdx.x+i*blockDim.x;
    average += (index < noReal) ? U[index+spikeNo*noReal] : 0.0f;
  }
  average = blockReduceSum( average);
  if (threadIdx.x==0) {
    U[spikeNo] = average/noReal;
  }
}

void circshift( float *w, int shift) {
  int i;
  float dummy[noThreads];
  # pragma unroll
  for (i=0;i<noThreads-shift;i++) {
    dummy[i] = w[shift+i];
  }
  # pragma unroll
  for (i=0;i<shift;i++) {
    dummy[noThreads-shift+i] = w[i];
  }
  # pragma unroll
  for (i=0;i<noThreads;i++) {
    w[i] = dummy[i];
  }
}

__device__ struct firing warpReduceMin( struct firing val) {
  float dummyTime;
  unsigned int dummyIndex;
  for (int offset = warpSize/2; offset>0; offset/=2) {
    dummyTime  = __shfl_down( val.time, offset);
    dummyIndex = __shfl_down( val.index, offset);
    val.time   = (val.time < dummyTime) ? val.time : dummyTime;
    val.index  = (val.time < dummyTime) ? val.index : dummyIndex;
  }
  return val;
}

__device__ struct firing blockReduceMin( struct firing val) {
  __shared__ struct firing shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceMin( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val.time  = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].time  : 0.0f;
  val.index = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].index : 0;

  if (wid==0) {
    val = warpReduceMin( val);
  }

  if (threadIdx.x==0) {
    shared[0] = val;
  }
  __syncthreads();

  return shared[0];
}


/* These functions are to help with doing reductions */
__device__ struct averaging warpReduceSum( struct averaging val) {
  for (int offset = warpSize/2; offset>0; offset/=2) {
    val.t   += __shfl_down( val.t, offset);
    val.x   += __shfl_down( val.x, offset);
    val.tSq += __shfl_down( val.tSq, offset);
    val.xt  += __shfl_down( val.xt, offset);
    val.count += __shfl_down( val.count, offset);
  }
  return val;
}

__device__ struct averaging blockReduceSum( struct averaging val) {
  __shared__ struct averaging shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceSumSimultaneous( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val.t     = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].t : 0.0f;
  val.x     = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].x : 0;
  val.tSq   = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].tSq : 0.0f;
  val.xt    = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].xt : 0.0f;
  val.count = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].count : 0;

  if (wid==0) {
    val = warpReduceSumSimultaneous( val);
  }

  return val;
}

