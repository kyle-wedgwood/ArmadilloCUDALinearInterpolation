#ifndef EVENTSTABILITYHEADERDEF
#define EVENTSTABILITYHEADERDEF

#include "AbstractStabilityClass.hpp"

class EventStability:
  public AbstractStability
{
  private:

    void ComputeDFDU(const arma::vec& u, arma::mat& jacobian);

};

#ifndef
