#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <armadillo>
#include <curand.h>
#include <assert.h>
#include "EventDrivenMap.hpp"
#include "parameters.hpp"

/* Error checking */
#define CUDA_ERROR_CHECK
#define CURAND_ERROR_CHECK
#define CUDA_CALL( err) __cudaCall( err, __FILE__, __LINE__ )
#define CURAND_CALL( err) __curandCall( err, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCall( cudaError err, const char *file, const int line )
{
  #ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
  #endif
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
  #ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
  #endif
}

inline void __curandCall( curandStatus_t err, const char *file, const int line)
{
#ifdef CURAND_ERROR_CHECK
  if ( CURAND_STATUS_SUCCESS != err)
  {
    fprintf( stderr, "curandCall() failed at %s:%i",
                 file, line);
    exit( -1 );
  }
  #endif
}

// Specialised constructor
EventDrivenMap::EventDrivenMap(const arma::vec* pParameters, unsigned int noReal)
{

  // Read in parameters
  mpHost_p = new arma::fvec((*pParameters).n_elem);
  *mpHost_p = arma::conv_to<arma::fvec>::from(*pParameters);

  // TEST
  printf("Host beta = %f\n",(*mpHost_p)[0]);

  // Declare memory for temporary storage of U and F
  mpU = new arma::fvec(noSpikes);
  mpF = new arma::fvec(noSpikes);

  // CUDA stuff
  mNoReal    = noReal;
  mNoThreads = noThreads;
  mNoBlocks  = (mNoReal+mNoThreads-1)/mNoThreads;

  // Set initial time horizon
  mFinalTime = timeHorizon;

  // allocate memory on CPU
  mpHost_lastSpikeInd = (unsigned short*) malloc( noSpikes*sizeof(short));

  // allocate memory on GPU
  CUDA_CALL( cudaMalloc( &mpDev_p, mpHost_p->n_elem*sizeof(float) ));
  CUDA_CALL( cudaMalloc( &mpDev_beta, mNoReal*mNoThreads*sizeof(float) ));
  CUDA_CALL( cudaMalloc( &mpDev_v, mNoReal*mNoThreads*sizeof(float) ));
  CUDA_CALL( cudaMalloc( &mpDev_s, mNoReal*mNoThreads*sizeof(float) ));
  CUDA_CALL( cudaMalloc( &mpDev_w, mNoThreads*sizeof(float) ));
  CUDA_CALL( cudaMalloc( &mpDev_U, (noSpikes+1)*sizeof(float) ));
  CUDA_CALL( cudaMalloc( &mpDev_lastSpikeInd,
        mNoReal*noSpikes*sizeof(unsigned short) ));
  CUDA_CALL( cudaMalloc( &mpDev_lastSpikeTime, mNoReal*noSpikes*sizeof(float)
        ));
  CUDA_CALL( cudaMalloc( &mpDev_crossedSpikeInd,
        mNoReal*noSpikes*sizeof(unsigned short) ));
  CUDA_CALL( cudaMalloc( &mpDev_crossedSpikeTime,
        mNoReal*noSpikes*sizeof(float) ));

  // Set up coupling kernel
  BuildCouplingKernel();

  // Copy parameters over
  CUDA_CALL( cudaMemcpy(mpDev_p,mpHost_p->begin(),(*mpHost_p).n_elem*sizeof(float),cudaMemcpyHostToDevice));

  // initialise random number generators
  CURAND_CALL( curandCreateGenerator( &mGen, CURAND_RNG_PSEUDO_DEFAULT));
  ResetSeed();
  mParStdDev = 0.0f;
}

void EventDrivenMap::BuildCouplingKernel()
{
  float *w;
  w = (float*) malloc( mNoThreads*sizeof(float));
  for (int i=0;i<mNoThreads;++i)
  {
    float x = -L + (float)(2*L/mNoThreads)*i;
    w[i] = (a1*exp(-b1*abs(x))-a2*exp(-b2*abs(x)))*2*L/mNoThreads;
  }
  circshift(w,mNoThreads/2);
  CUDA_CALL( cudaMemcpy(mpDev_w,w,mNoThreads*sizeof(float),cudaMemcpyHostToDevice));
  FILE *fp = fopen("test.dat","w");
  for (int i=0;i<mNoThreads;++i)
  {
    fprintf(fp,"%f\n",w[i]);
  }
  fclose(fp);
  free(w);
}

EventDrivenMap::~EventDrivenMap()
{
  delete mpU;
  delete mpF;
  delete mpHost_p;

  free(mpHost_lastSpikeInd);

  cudaFree(mpDev_p);
  cudaFree(mpDev_beta);
  cudaFree(mpDev_v);
  cudaFree(mpDev_s);
  cudaFree(mpDev_w);
  cudaFree(mpDev_U);
  cudaFree(mpDev_lastSpikeInd);
  cudaFree(mpDev_lastSpikeTime);
  cudaFree(mpDev_crossedSpikeInd);
  cudaFree(mpDev_crossedSpikeTime);

  curandDestroyGenerator(mGen);
}

void EventDrivenMap::ComputeF(const arma::vec& Z, arma::vec& f)
{

  arma::vec U0(noSpikes+1);
  arma::vec UT(noSpikes);
  arma::fvec fU(noSpikes+1);

  // First generate initial spike indices
  initialSpikeInd( Z);
  printf("Initial spike indices = \n");
  for (int i=0;i<noSpikes;++i)
  {
    printf("%d\n",mpHost_lastSpikeInd[i]);
  }
  unsigned short* mpHost_lastSpikeIndBig;
  mpHost_lastSpikeIndBig = (unsigned short*) malloc(
      mNoReal*noSpikes*sizeof(short));
  CUDA_CALL( cudaMemcpy( mpHost_lastSpikeIndBig, mpDev_lastSpikeInd,
        mNoReal*noSpikes*sizeof(short), cudaMemcpyDeviceToHost));
  FILE *fp = fopen("testInitLastSpikeInd.dat","w");
  for (int i=0;i<mNoReal*noSpikes;++i)
  {
    fprintf(fp,"%f\n",(float) mpHost_lastSpikeIndBig[i]);
  }
  fclose(fp);

  // Then, put vector in correct form
  ZtoU(Z,U0);

  printf("U0 = \n");
  std::cout << U0 << std::endl;

  // Then, typecast data as floats
  fU = arma::conv_to<arma::fvec>::from(U0);

  printf("Initial fU = \n");
  std::cout << fU << std::endl;

  // Assuming that weight kernel does not change
  CUDA_CALL( cudaMemcpy(mpDev_U,fU.begin(),(noSpikes+1)*sizeof(float),cudaMemcpyHostToDevice));

  // Introduce parameters heterogeneity
  CURAND_CALL( curandGenerateNormal( mGen, mpDev_beta, mNoReal*mNoThreads, (*mpHost_p)[0], mParStdDev));

  // Lift - working
  LiftKernel<<<mNoReal,mNoThreads>>>(mpDev_s,mpDev_v,mpDev_p,mpDev_U,mNoReal);
  CUDA_CHECK_ERROR();
  float *mpHost_Lift;
  mpHost_Lift = (float*) malloc( 2*mNoThreads*sizeof(float));
  CUDA_CALL( cudaMemcpy( mpHost_Lift,mpDev_v,mNoThreads*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CALL( cudaMemcpy( mpHost_Lift+mNoThreads,mpDev_s,mNoThreads*sizeof(float),cudaMemcpyDeviceToHost));

  fp = fopen("testLift.dat","w");
  for (int i=0;i<mNoThreads;++i)
  {
    fprintf(fp,"%f\t%f\n",mpHost_Lift[i],mpHost_Lift[i+mNoThreads]);
  }
  fclose(fp);
  printf("First line of lift: %f \t %f \n",mpHost_Lift[0],mpHost_Lift[mNoThreads]);
  free(mpHost_Lift);

  // Copy data to GPU
  //cudaMemcpy( dev_w, w, mNoThreads*sizeof(float), cudaMemcpyHostToDevice );

  // Evolve
  printf("Time horizon = %f\n",mFinalTime);
  float par_val;
  CUDA_CALL( cudaMemcpy(&par_val,mpDev_p,sizeof(float),cudaMemcpyDeviceToHost));
  printf("Device beta = %f\n",par_val);
  EvolveKernel<<<mNoReal,mNoThreads>>>(mpDev_v,mpDev_s,mpDev_beta,mpDev_w,mFinalTime,mpDev_lastSpikeInd,mpDev_lastSpikeTime,
      mpDev_crossedSpikeInd,mpDev_crossedSpikeTime,mNoReal);
  CUDA_CHECK_ERROR();
  printf("Evolve kernel started.\n");

  unsigned short *mpHost_spikeInd;
  float *mpHost_spikeTime;
  mpHost_spikeInd = (unsigned short*) malloc( noSpikes*mNoReal*sizeof(short));
  mpHost_spikeTime = (float*) malloc( noSpikes*mNoReal*sizeof(float));

  char filename1[] = "testLastSpikeInd.dat";
  char filename2[] = "testLastSpikeTime.dat";
  char filename3[] = "testCrossedSpikeInd.dat";
  char filename4[] = "testCrossedSpikeTime.dat";

  CUDA_CALL( cudaMemcpy( mpHost_spikeInd, mpDev_lastSpikeInd, noSpikes*mNoReal*sizeof(unsigned short), cudaMemcpyDeviceToHost));
  CUDA_CALL( cudaMemcpy( mpHost_spikeTime, mpDev_lastSpikeTime, noSpikes*mNoReal*sizeof(float), cudaMemcpyDeviceToHost));

  SaveData(noSpikes*mNoReal,mpHost_spikeTime,filename2);
  for (int i=0;i<noSpikes*mNoReal;i++)
  {
    mpHost_spikeTime[i] = (float)mpHost_spikeInd[i];
  }
  SaveData(noSpikes*mNoReal,mpHost_spikeTime,filename1);

  CUDA_CALL( cudaMemcpy( mpHost_spikeInd, mpDev_crossedSpikeInd, noSpikes*mNoReal*sizeof(unsigned short), cudaMemcpyDeviceToHost));
  CUDA_CALL( cudaMemcpy( mpHost_spikeTime, mpDev_crossedSpikeTime, noSpikes*mNoReal*sizeof(float), cudaMemcpyDeviceToHost));

  SaveData(noSpikes*mNoReal,mpHost_spikeTime,filename4);
  for (int i=0;i<noSpikes*mNoReal;i++)
  {
    mpHost_spikeTime[i] = (float)mpHost_spikeInd[i];
  }
  SaveData(noSpikes*mNoReal,mpHost_spikeTime,filename3);

  free(mpHost_spikeInd);
  free(mpHost_spikeTime);

  // Restrict
  RestrictKernel<<<noSpikes*mNoReal,mNoThreads>>>( mpDev_lastSpikeTime, mpDev_lastSpikeInd,
      mpDev_crossedSpikeTime, mpDev_crossedSpikeInd, mFinalTime, mNoReal);
  CUDA_CHECK_ERROR();

  char filename5[] = "testAverages.dat";
  float* mpHost_averages;
  mpHost_averages = (float*) malloc( noSpikes*mNoReal*sizeof(float));
  cudaMemcpy( mpHost_averages, mpDev_lastSpikeTime, noSpikes*mNoReal*sizeof(float), cudaMemcpyDeviceToHost);
  SaveData( noSpikes*mNoReal,mpHost_averages,filename5);
  free(mpHost_averages);

  realisationReductionKernelBlocks<<<noSpikes,mNoThreads>>>(
      mpDev_U, mpDev_lastSpikeTime, mNoReal);
  CUDA_CHECK_ERROR();

  printf("Averaging done.\n");

  // Copy data back to CPU
  fU.resize(noSpikes);
  CUDA_CALL( cudaMemcpy( fU.begin(), mpDev_U, noSpikes*sizeof(float), cudaMemcpyDeviceToHost ));

  printf("Final fU = \n");
  std::cout << fU << std::endl;

  // Compute F
  UT = arma::conv_to<arma::vec>::from(fU);

  f = -U0[0]*U0.rows(1,noSpikes) - UT + U0[0]*mFinalTime;
  printf("f = \n");
  std::cout << f << std::endl;
}

void EventDrivenMap::SetTimeHorizon( const float T)
{
  assert(T>0);
  mFinalTime = T;
}

void EventDrivenMap::SetNoRealisations( const int noReal)
{
  assert(noReal>0);
  mNoReal = noReal;
}

void EventDrivenMap::SetParameterStdDev( const float sigma)
{
  assert(sigma>=0);
  mParStdDev = sigma;
}

void EventDrivenMap::SetParameters( const unsigned int parId, const float parVal)
{
  assert(parId<=(*mpHost_p).n_elem);
  (*mpHost_p)[parId] = parVal;
  CUDA_CALL( cudaMemcpy(mpDev_p+parId,&parVal,sizeof(float),cudaMemcpyHostToDevice));
}

void EventDrivenMap::ResetSeed()
{
  CURAND_CALL( curandSetPseudoRandomGeneratorSeed( mGen, (unsigned long long) clock() ));
}

void EventDrivenMap::initialSpikeInd( const arma::vec& U)
{
  unsigned int i,m;
  mpHost_lastSpikeInd[0] = mNoThreads/2;
  for (m=1;m<noSpikes;m++) {
    for (i=mpHost_lastSpikeInd[m-1];i>0;i--) {
      if (-L+(float)(2*i*L/mNoThreads)<-U[0]*U[m]) {
        mpHost_lastSpikeInd[m] = i;
        break;
      }
    }
  }
  CUDA_CALL( cudaMemcpy2D( mpDev_lastSpikeInd, mNoReal*sizeof(short),
        mpHost_lastSpikeInd, sizeof(short), sizeof(short), noSpikes, cudaMemcpyHostToDevice ));
  initialSpikeIndCopyKernel<<<(mNoReal*noSpikes+mNoThreads-1)/mNoThreads,mNoThreads>>>( mpDev_lastSpikeInd, mNoReal);
}

__global__ void initialSpikeIndCopyKernel( unsigned short* pLastSpikeInd, const unsigned int noReal)
{
  unsigned int globalIndex = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int spikeNo = globalIndex / noReal;
  if (globalIndex<noReal*noSpikes)
  {
    pLastSpikeInd[globalIndex] = pLastSpikeInd[spikeNo*noReal];
  }
}

void EventDrivenMap::ZtoU( const arma::vec& Z, arma::vec& U)
{
  assert(U.n_elem==Z.n_elem+1);
  U[0] = Z[0];
  U[1] = 0.0;
  for (int i=2;i<=noSpikes;i++) {
    U[i] = Z[i-1];
  }
}

void EventDrivenMap::UtoZ( const arma::vec *U, arma::vec *Z)
{
  Z[0] = U[0];
  for (int i=1;i<noSpikes;i++) {
    Z[i] = U[i+1];
  }
}

__global__ void LiftKernel( float *S, float *v, const float *par, const float *U,
    const unsigned int noReal)
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

__device__ float fun( float t, float v, float s, float beta)
{
  return v*exp(-t)+I*(1.0f-exp(-t))+s*exp(-t)/(1.0f-beta)*(exp((1.0f-beta)*t)-1.0f)-vth;
}

__device__ float dfun( float t, float v, float s, float beta)
{
  return I*exp(-t)-v*exp(-t)+s*exp(-t)*exp(-t*(beta-1))+(s*exp(-t)*(exp(-t*(beta-1))-1.0f))/(beta-1);
}

__device__ float eventTime( float v0, float s0, float beta)
{
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

__global__ void EvolveKernel( float *v, float *s, const float *beta,
    const float *w, const float finalTime, unsigned short *global_lastSpikeInd,
    float *global_lastSpikeTime, unsigned short *global_crossedSpikeInd,
    float *global_crossedSpikeTime, unsigned int noReal)
{
  __shared__ unsigned short local_lastSpikeInd[noSpikes];
  __shared__ unsigned short local_crossedSpikeInd[noSpikes];
  __shared__ float local_lastSpikeTime[noSpikes];
  __shared__ float local_crossedSpikeTime[noSpikes];
  __shared__ unsigned int noCrossed;
  __shared__ bool crossings[noSpikes];
  float currentTime = 0.0f;
  float local_v, local_s, local_beta;
  unsigned short minIndex;
  struct EventDrivenMap::firing val;

  // load values from global memory
  local_v = v[threadIdx.x+blockIdx.x*blockDim.x];
  local_s = s[threadIdx.x+blockIdx.x*blockDim.x];
  local_beta = beta[threadIdx.x+blockIdx.x*blockDim.x];

  if (threadIdx.x<noSpikes)
  {
    local_lastSpikeInd[threadIdx.x] =
      global_lastSpikeInd[threadIdx.x*noReal+blockIdx.x];
    crossings[threadIdx.x] = false;
  }
  noCrossed = 0;
  while (noCrossed<noSpikes)
  {
    // find next firing times
    val.time  = eventTime(local_v,local_s,local_beta);
    val.index = threadIdx.x;

    // perform reduction to find minimum spike time
    val = blockReduceMin( val);

    // val now contains minimum spike time and index
    // update values to spike time
    local_v *= exp(-val.time);
    local_v +=
      I*(1.0f-exp(-val.time))+local_s*exp(-val.time)/(1.0f-local_beta)*(exp((1.0f-local_beta)*val.time)-1.0f);
    local_v *= (threadIdx.x!=val.index);
    local_s *= exp(-local_beta*val.time);
    local_s += local_beta*w[(threadIdx.x-val.index)*(threadIdx.x>=val.index)+(val.index-threadIdx.x)*(threadIdx.x<val.index)];

    currentTime += val.time;

    // store values
    if (threadIdx.x==0)
    {
      // First calculate which crossing spike belongs to
      minIndex = 0;
      for (int i=1;i<noSpikes;i++)
      {
        minIndex += ((std::abs((int)(val.index-local_lastSpikeInd[i])))<(std::abs((int)(val.index-local_lastSpikeInd[minIndex]))));
      }
      if (!crossings[minIndex])
      {
        if (currentTime>finalTime)
        {
          local_crossedSpikeTime[minIndex] = currentTime;
          local_crossedSpikeInd[minIndex]  = val.index;
          crossings[minIndex] = true;
          noCrossed += 1;
        }
        else
        {
          local_lastSpikeTime[minIndex] = currentTime;
          local_lastSpikeInd[minIndex]  = val.index;
        }
      }
    }
  }

  /*
  // Save into global memory
  if (threadIdx.x<noSpikes)
  {
    global_lastSpikeInd[blockIdx.x*noSpikes+threadIdx.x]     = local_lastSpikeInd[threadIdx.x];
    global_lastSpikeTime[blockIdx.x*noSpikes+threadIdx.x]    = local_lastSpikeTime[threadIdx.x];
    global_crossedSpikeInd[blockIdx.x*noSpikes+threadIdx.x]  = local_crossedSpikeInd[threadIdx.x];
    global_crossedSpikeTime[blockIdx.x*noSpikes+threadIdx.x] = local_crossedSpikeTime[threadIdx.x];
  }
  */

  // Save into global memory
  if (threadIdx.x<noSpikes)
  {
    global_lastSpikeInd[blockIdx.x+threadIdx.x*noReal]     =
      local_lastSpikeInd[threadIdx.x];
    global_lastSpikeTime[blockIdx.x+threadIdx.x*noReal]    =
      local_lastSpikeTime[threadIdx.x];
    global_crossedSpikeInd[blockIdx.x+threadIdx.x*noReal]  =
      local_crossedSpikeInd[threadIdx.x];
    global_crossedSpikeTime[blockIdx.x+threadIdx.x*noReal] =
      local_crossedSpikeTime[threadIdx.x];
  }
}

/*
__global__ void EvolveKernel( float *v, float *s, const float *beta,
    const float *w, const float finalTime, unsigned short *global_lastSpikeInd,
    float *global_lastSpikeTime, unsigned short *global_crossedSpikeInd,
    float *global_crossedSpikeTime, unsigned int noReal)
{
  __shared__ unsigned short local_lastSpikeInd[noSpikes];
  __shared__ unsigned short local_crossedSpikeInd[noSpikes];
  __shared__ float local_lastSpikeTime[noSpikes];
  __shared__ float local_crossedSpikeTime[noSpikes];
  __shared__ unsigned int noCrossed;
  float currentTime = 0.0f;
  float local_v, local_s, local_beta;
  unsigned short minIndex;
  struct EventDrivenMap::firing val;

  // load values from global memory
  local_v = v[threadIdx.x+blockIdx.x*blockDim.x];
  local_s = s[threadIdx.x+blockIdx.x*blockDim.x];
  local_beta = beta[threadIdx.x+blockIdx.x*blockDim.x];

  if (threadIdx.x<noSpikes)
  {
    local_lastSpikeInd[threadIdx.x] = global_lastSpikeInd[blockIdx.x*noSpikes+threadIdx.x];
  }
  noCrossed = 0;
  while (noCrossed<(1<<noSpikes)-1)
  {
    // find next firing times
    val.time  = eventTime(local_v,local_s,local_beta);
    val.index = threadIdx.x;

    // perform reduction to find minimum spike time
    val = blockReduceMin( val);

    // val now contains minimum spike time and index
    // update values to spike time
    local_v *= exp(-val.time);
    local_v +=
      I*(1.0f-exp(-val.time))+local_s*exp(-val.time)/(1.0f-local_beta)*(exp((1.0f-local_beta)*val.time)-1.0f);
    local_v *= (threadIdx.x!=val.index);
    local_s *= exp(-local_beta*val.time);
    local_s += local_beta*w[(threadIdx.x-val.index)*(threadIdx.x>=val.index)+(val.index-threadIdx.x)*(threadIdx.x<val.index)];

    currentTime += val.time;

    // store values
    if (threadIdx.x==0)
    {
      // First calculate which crossing spike belongs to
      minIndex = 0;
      for (int i=1;i<noSpikes;i++)
      {
        minIndex += ((std::abs((int)(val.index-local_lastSpikeInd[i])))<(std::abs((int)(val.index-local_lastSpikeInd[minIndex]))));
      }
      if (!(noCrossed & (1<<minIndex)))
      {
        if (currentTime>finalTime)
        {
          local_crossedSpikeTime[minIndex] = currentTime;
          local_crossedSpikeInd[minIndex]  = val.index;
          noCrossed += (1<<minIndex);
        }
        else
        {
          local_lastSpikeTime[minIndex] = currentTime;
          local_lastSpikeInd[minIndex]  = val.index;
        }
      }
    }
  }

  // Save into global memory
  if (threadIdx.x<noSpikes)
  {
    global_lastSpikeInd[blockIdx.x*noSpikes+threadIdx.x]     = local_lastSpikeInd[threadIdx.x];
    global_lastSpikeTime[blockIdx.x*noSpikes+threadIdx.x]    = local_lastSpikeTime[threadIdx.x];
    global_crossedSpikeInd[blockIdx.x*noSpikes+threadIdx.x]  = local_crossedSpikeInd[threadIdx.x];
    global_crossedSpikeTime[blockIdx.x*noSpikes+threadIdx.x] = local_crossedSpikeTime[threadIdx.x];
  }

  // Save into global memory
  if (threadIdx.x==2)
  {
    global_lastSpikeInd[blockIdx.x+threadIdx.x*noReal]     = local_lastSpikeInd[threadIdx.x];
    global_lastSpikeTime[blockIdx.x+threadIdx.x*noReal]    = local_lastSpikeTime[threadIdx.x];
    global_crossedSpikeInd[blockIdx.x+threadIdx.x*noReal]  = local_crossedSpikeInd[threadIdx.x];
    global_crossedSpikeTime[blockIdx.x+threadIdx.x*noReal] = local_crossedSpikeTime[threadIdx.x];
  }
}
*/

/* Restrict functions */
__global__ void RestrictKernel( float *global_lastSpikeTime,
                                const unsigned short *global_lastSpikeInd,
                                const float *global_crossedSpikeTime,
                                const unsigned short *global_crossedSpikeInd,
                                const float finalTime,
                                const unsigned int noReal)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  if (index<noSpikes*noReal)
  {
    float t0 = global_lastSpikeTime[index];
    float t1 = global_crossedSpikeTime[index];
    float x0 = -L+2.0f*L/noThreads*global_lastSpikeInd[index];
    float x1 = -L+2.0f*L/noThreads*global_crossedSpikeInd[index];
    global_lastSpikeTime[index] = x0+(finalTime-t0)*(x1-x0)/(t1-t0);
  }
}

__global__ void realisationReductionKernelBlocks( float *V,
                                                  const float *U,
                                                  const unsigned int noReal)
{
  unsigned int i, spikeNo = blockIdx.x;
  unsigned int index;
  unsigned int noLoad = (noReal+blockDim.x-1)/blockDim.x;
  float average = 0.0f;

  for (i=0;i<noLoad;i++) {
    index = threadIdx.x+i*blockDim.x;
    average += (index < noReal) ? U[index+spikeNo*noReal] : 0.0f;
    //average += (index < noReal) ? U[noSpikes*index+spikeNo] : 0.0f;
  }
  average = blockReduceSum( average);
  if (threadIdx.x==0) {
    V[spikeNo] = average/noReal;
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

__device__ struct EventDrivenMap::firing warpReduceMin( struct EventDrivenMap::firing val) {
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

__device__ struct EventDrivenMap::firing blockReduceMin( struct EventDrivenMap::firing val) {
  __shared__ struct EventDrivenMap::firing shared[32];
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
  val = shared[0];

  return val;
}

__device__ float warpReduceSum( float val) {
  for (int offset = warpSize/2; offset>0; offset/=2) {
    val += __shfl_down( val, offset);
  }
  return val;
}

__device__ float blockReduceSum( float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceSum( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : 0.0f;

  if (wid==0) {
    val = warpReduceSum( val);
  }

  return val;
}

void SaveData( int npts, float *x, char *filename) {
  FILE *fp = fopen(filename,"w");
  for (int i=0;i<npts;i++) {
    fprintf(fp,"%f\n",x[i]);
  }
  fclose(fp);
}
