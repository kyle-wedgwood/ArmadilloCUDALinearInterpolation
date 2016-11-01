#ifndef EVENTDRIVEMAPHEADERDEF
#define EVENTDRIVEMAPHEADERDEF

#include <cmath>
#include <armadillo>
#include <curand.h>
#include <cassert>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"

class EventDrivenMap:
  public AbstractNonlinearProblem
{

  public:

    // Specialised constructor
    EventDrivenMap( const arma::vec* pParameters, unsigned int noReal);

    // Destructor
    ~EventDrivenMap();

    // Right-hand side
    void ComputeF( const arma::vec& u, arma::vec& f);

    // Overload function to accept utilde
    void ComputeF( const arma::vec& u, arma::vec& f, const arma::vec& uTilde);

    // Equation-free stuff
    void SetTimeHorizon( const double T);

    // CUDA stuff
    // Change number of realisations
    void SetNoRealisations( const int noReal);

    void SetNoThreads( const int noThreads);

    // Set variance
    void SetParameterStdDev( const double sigma);

    // Set parameter
    void SetParameters( const unsigned int parId, const double parVal);

    // Reset seed
    void ResetSeed();

    // Set new seed
    void SetNewSeed();

    // Post process data
    void PostProcess();

    // Toggle debug flag
    void SetDebugFlag( const bool val);

    // Structure to store firing times and indices */
    struct __align__(16) firing{
      double time;
      unsigned int index;
    };

  private:

    // Hiding default constructor
    EventDrivenMap();

    // double vector for temporary storage
    arma::fvec* mpU;
    arma::fvec* mpF;

    // double vector for parameters
    arma::fvec* mpHost_p;

    // Integration time
    double mFinalTime;

    // threads & blocks
    unsigned int mNoReal;
    unsigned int mNoThreads;
    unsigned int mNoBlocks;
    unsigned int mNoSpikes;

    // CPU variables
    unsigned short *mpHost_lastSpikeInd;

    // GPU variables
    double *mpDev_p;
    double *mpDev_beta;
    double *mpDev_v;
    double *mpDev_s;
    double *mpDev_w;
    double *mpDev_U;
    double *mpDev_lastSpikeTime;
    double *mpDev_crossedSpikeTime;
    unsigned short *mpDev_lastSpikeInd;
    unsigned short *mpDev_crossedSpikeInd;
    unsigned int *mpDev_accept;

    curandGenerator_t mGen; // random number generator
    unsigned long long mSeed; // seed for RNG
    double mParStdDev;

    // Functions to do lifting
    void initialSpikeInd( const arma::vec& U);

    void ZtoU( const arma::vec& Z,
               arma::vec& U,
               const arma::vec& Ztilde);

    void UtoZ( const arma::vec *U, arma::vec *Z);

    void BuildCouplingKernel();

    // For debugging purposes
    bool mDebugFlag;

    void SaveInitialSpikeInd();

    void SaveLift();

    void SaveEvolve();

    void SaveRestrict();

    void SaveAveraged();
};

__global__ void LiftKernel( double *s, double *v, const double *par, const double *U,
    const unsigned int noReal);

// Functions to find spike time
__device__ double fun( double t, double v, double s, double beta);

__device__ double dfun( double t, double v, double s, double beta);

__device__ double eventTime( double v0, double s0, double beta);

// evolution
__global__ void EvolveKernel( double *v, double *s, const double *beta,
    const double *w, const double finalTime, unsigned short *global_lastSpikeInd,
    double *global_lastFiringTime, unsigned short *global_crossedSpikeInd,
    double *global_crossedFiringTime, unsigned int *global_accept, unsigned int noReal);

// restriction
__global__ void RestrictKernel( double *global_lastSpikeTime,
                                const unsigned short *global_lastSpikeInd,
                                const double *global_crossedSpikeTime,
                                const unsigned short *global_crossedSpikeInd,
                                const double finalTime,
                                const unsigned int noReal);

// count number of active realisations
__global__ void CountRealisationsKernel( unsigned int *accept, const unsigned int noReal);

// averaging functions
__global__ void realisationReductionKernelBlocks( double *dev_V,
                                                  const double *dev_U,
                                                  const unsigned int noReal,
                                                  const unsigned int *accept);

// helper functions
__global__ void initialSpikeIndCopyKernel( unsigned short* pLastSpikeInd, const unsigned int noReal);

void circshift( double *w, int shift, unsigned int noThreads);
__device__ struct EventDrivenMap::firing warpReduceMin( struct EventDrivenMap::firing val);
__device__ struct EventDrivenMap::firing blockReduceMin( struct EventDrivenMap::firing val);
__device__ double warpReduceSum ( double val);
__device__ double blockReduceSum( double val);
__device__ int warpReduceSumInt ( int val);
__device__ int blockReduceSumInt( int val);

void SaveData( int npts, double *x, char *filename);

#endif
