#include <iostream>
#include <iomanip>
#include <armadillo>
#include "EventStability.hpp"

void EventStability::ComputeDFDU(const arma::vec& u, arma::mat& jacobian)
{

  // Perturbed solution
  arma::vec uTilde(size(u),arma::fill::zeros);

  // Problem size
  int problem_size = u.n_rows;

  // Perturbed residual
  arma::vec f(problem_size);
  arma::vec df(problem_size);

  // Epsilon
  double epsilon = mFiniteDifferenceEpsilon;

  mpProblem->ComputeF(u,f);

  // For each column
  for (int i=0; i<problem_size; i++)
  {
    // Restore original solution, then perturb it
    if (i>0)
    {
      uTilde(i-1) = 0.0;
    }
    uTilde(i) = epsilon;

    // Perturbed residual
    mpProblem->ComputeF(u,df,uTilde);

    // Assign jacobian column
    jacobian.col(i) = (df - f) * pow(epsilon,-1);
  }

}