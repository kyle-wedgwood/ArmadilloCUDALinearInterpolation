#ifndef CUDANONLINEARPROBLEMHEADERDEF
#define CUDANONLINEARPROBLEMHEADERDEF

#include <curand.h>
#include <time.h>

class CUDANonlinearProblem
{

  public:

    //Specialised constructor
    CUDANonlinearProblem(const int noReal, const float *pu, const float *mpParameters);

    // Destructor
    ~CUDANonlinearProblem();

    // Set number of threads (and blocks)
    void SetNoThreads( const int number);

    // Change number of realisations
    void SetNoRealisations( const int noReal);

    // Set variance
    void SetVariance( const float variance);

    // Reset seed
    void ResetSeed();

  private:

    curandGenerator_t gen;

    unsigned int N;

    float sigma;
    unsigned int noThreads;
    unsigned int noBlocks;

    // Hiding default constructor
    CUDANonlinearProblem();

};

void cudaSum( float *val, const int N, const unsigned int noThreads);

__global__ void local_f( const float *dev_pPars, const float *dev_pf, int N);

__device__ void warpReduceSum( float *val);

__device__ void blockReduceSum( float *val);

__global__ void deviceReduceKernel(float *in, float *out, int N);

#endif
