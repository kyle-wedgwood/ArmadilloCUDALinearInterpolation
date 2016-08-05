#include <iostream>
#include <cstdlib>
#include <cmath>
#include <armadillo>

#include "NewtonSolver.hpp"
#include "Stability.hpp"
#include "EventDrivenMap.hpp"
#include "parameters.hpp"

int main(int argc, char* argv[])
{

  // Parameters
  arma::vec* p_parameters = new arma::vec(1);
  (*p_parameters) << 13.0589f;

  // Instantiate problem
  unsigned int noReal = 1000;
  EventDrivenMap* p_event = new EventDrivenMap(p_parameters,noReal);
  p_event->SetParameterStdDev(0.5);

  // Initial guess
  arma::vec* p_solution_old = new arma::vec(noSpikes);
  (*p_solution_old) << 0.3310f << 0.6914f << 1.3557f;

  // Perturb solution
  double sigma = 0.01;
  arma::vec perturbation = arma::vec(noSpikes,arma::fill::randn);
  perturbation = arma::normalise(perturbation,1);
  (*p_solution_old) += sigma*perturbation;

  // Newton solver parameter list
  NewtonSolver::ParameterList pars;
  pars.tolerance = 0.0;
  pars.maxIterations = 10;
  pars.printOutput = true;
  pars.damping = 0.2;
  pars.finiteDifferenceEpsilon = 1e-3;

  // Instantiate newton solver (finite differences)
  NewtonSolver* p_newton_solver = new NewtonSolver(p_event, p_solution_old, &pars);

  // Solve
  arma::vec* p_solution_new = new arma::vec(noSpikes);
  arma::vec* p_residual_history = new arma::vec(); // size assigned by Newton solver
  AbstractNonlinearSolver::ExitFlagType exitFlag;

  // Vary number of neurons (threads)
  arma::vec noThreadsVector(3);
  noThreadsVector << (1<<10) << (1<<9) << (1<<8);

  // Prepare Newton solver
  p_newton_solver->SetInitialGuess(p_solution_old);

  // Matrix to store data
  arma::mat* p_data = new arma::mat(pars.maxIterations+1,noThreadsVector.n_elem+1);
  (*p_data).col(0) = arma::linspace(0,pars.maxIterations,pars.maxIterations+1);

  // Now loop over steps
  for (int i=0;i<noThreadsVector.n_elem;++i)
  {
    p_event->SetNoThreads(noThreadsVector(i));
    p_newton_solver->Solve(*p_solution_new,*p_residual_history,exitFlag);
    (*p_data).col(i+1) = *p_residual_history;
  }

  // Save data
  p_data->save("ResidualsVaryN.dat",arma::arma_ascii);

  // Clean
  delete p_parameters;
  delete p_event;
  delete p_solution_old;
  delete p_solution_new;
  delete p_residual_history;
  delete p_newton_solver;
  delete p_data;
}
