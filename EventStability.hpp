#ifndef EVENTSTABILITYHEADERDEF
#define EVENTSTABILITYHEADERDEF

#include "AbstractStabilityClass.hpp"

class EventStability:
  public AbstractStability
{
  public:

    EventStability( ProblemType type, AbstractNonlinearProblem* pProblem) :
      AbstractStability( type, pProblem) {};

    EventStability( ProblemType type,
                    AbstractNonlinearProblem* pProblem,
                    AbstractNonlinearProblemJacobian* pProblemJacobian) :
      AbstractStability( type, pProblem, pProblemJacobian) {};

  private:

    void ComputeDFDU(const arma::vec& u, arma::mat& jacobian);

};

#endif
