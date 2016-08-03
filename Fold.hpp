#ifndef FOLDHEADERDEF
#define FOLDHEADERDEF

#include <iostream>
#include <armadillo>
#include <curand.h>
#include <cassert>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"

class Fold:
  public AbstractNonlinearProblem,
  public AbstractNonlinearProblemJacobian
{

  public:

    //Specialised constructor
    Fold(const arma::vec* pParameters);

    // Destructor
    ~Fold();

    // Right-hand side
    void ComputeF(const arma::vec& u, arma::vec& f);

    // Jacobian
    void ComputeDFDU(const arma::vec& u, arma::mat& dfdu);

    // CUDA stuff
    // Set number of threads (and blocks)
    void SetNoThreads( const int number);

    // Change number of realisations
    void SetNoRealisations( const int noReal);

    // Set variance
    void SetVariance( const float variance);

    // Reset seed
    void ResetSeed();

  private:

    // Hiding default constructor
    Fold();

    // Parameters
    const arma::vec* mpParameters;

    // CUDA stuff
    curandGenerator_t gen;
    float sigma;

    unsigned int N;
    unsigned int nDim = 2;

    float *dev_p_pars;
    float *dev_p_f;
    float *dev_p_u;

    unsigned int noThreads;
    unsigned int noBlocks;

};

__device__ float warpReduceSum( float val);

__device__ float blockReduceSum( float val);

__global__ void deviceReduceKernel(float *in, float *out, int N, const unsigned int nDim);

__global__ void deviceReduceKernel(float *in, float *out, int N, const unsigned int nDim, const unsigned int offset);

void cudaSum( float *val, int N, const unsigned int nDim);

void cudaSumAlt( float *val, int N, const unsigned int nDim);

__global__ void local_f( const float *dev_pPars, const float *dev_pU, float *dev_pf, int N, unsigned int nDim);

#endif
