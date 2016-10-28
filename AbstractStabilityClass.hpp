#ifndef STABILITYHEADERDEF
#define STABILITYHEADERDEF

#include <cassert>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"

class AbstractStability
{
  public:

    // Problem type
    enum class ProblemType
    {
      flow,
      map,
      equationFree
    };

    // Constructor (Jacobian computed by finite differences explicitly)
    AbstractStability( ProblemType type, AbstractNonlinearProblem* pProblem);

    // Constructor (Jacobian passed by user)
    AbstractStability( ProblemType type,
                       AbstractNonlinearProblem* pProblem,
                       AbstractNonlinearProblemJacobian *pProblemJacobian);

    ~AbstractStability();

    int ComputeNumUnstableEigenvalues(const arma::cx_vec& eigenvalues);

    int ComputeNumUnstableEigenvalues(const arma::vec& u);

    int ComputeNumUnstableEigenvalues(arma::mat& jacobian);

    arma::cx_vec ComputeEigenvalues(const arma::vec& u);

    arma::cx_vec ComputeEigenvalues(arma::mat& jacobian);

  private:

    // Hiding default constructor
    AbstractStability();

    // Compute Jacobian via finite differences
    virtual void ComputeDFDU(const arma::vec& u, arma::mat& jacobian);

    // Problem interface
    AbstractNonlinearProblem* mpProblem;
    AbstractNonlinearProblemJacobian* mpProblemJacobian;
    ProblemType mProblemType;

    // Other parameters
    double mFiniteDifferenceEpsilon;

};

#endif
