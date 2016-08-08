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
  unsigned int noReal = 500;
  EventDrivenMap* p_event = new EventDrivenMap(p_parameters,noReal);

  // Initial guess
  arma::vec* p_solution_old = new arma::vec(noSpikes);
  (*p_solution_old) << 0.3310f << 0.6914f << 1.3557f;

  // Newton solver parameter list
  NewtonSolver::ParameterList pars;
  pars.tolerance = 5e-4;
  pars.maxIterations = 100;
  pars.printOutput = true;
  pars.damping = 0.2;

  // Instantiate newton solver (passing Jacobian)
  NewtonSolver* p_newton_solver = new NewtonSolver(p_event, p_solution_old, &pars);

  // Instantiate newton solver (finite differences)
  pars.finiteDifferenceEpsilon = 1e-2;

  // Solve
  arma::vec* p_solution_new = new arma::vec(noSpikes);
  arma::vec* p_residual_history = new arma::vec(); // size assigned by Newton solver
  AbstractNonlinearSolver::ExitFlagType exitFlag;
  std::cout << "\n ******** Solve using finite differences" << std::endl;

  // For computing eigenvalues
  Stability* p_stability = new Stability(Stability::ProblemType::equationFree,p_event);
  arma::mat* p_jacobian = new arma::mat(noSpikes,noSpikes);
  arma::cx_vec* p_eigenvalues = new arma::cx_vec(noSpikes);
  arma::vec* p_real_eigenvalues = new arma::vec(noSpikes);
  int N_steps = 100;
  int numUnstableEigenvalues = -1;

  // Add some heterogeneity
  float sigma = 0.1f;
  p_event->SetParameterStdDev(sigma*(*p_parameters)(0));
  printf("Setting parameter standard deviation to %f\n",sigma);

  // Save data
  //arma::mat* p_data = new arma::mat(N_steps,noSpikes+4,arma::fill::zeros);
  arma::mat* p_data = new arma::mat(0,noSpikes+4,arma::fill::zeros);

  // Now loop over steps
  double ds = -0.2;

  for (int i=0;i<N_steps;++i)
  {
    p_newton_solver->SetInitialGuess(p_solution_old);
    p_newton_solver->Solve(*p_solution_new,*p_residual_history,exitFlag,p_jacobian);

    /*
    numUnstableEigenvalues = p_stability->ComputeNumUnstableEigenvalues(*p_jacobian);
    std::cout << "Number of unstable eigenvalues = " << numUnstableEigenvalues << std::endl;

    if (numUnstableEigenvalues > 0)
    {
      std::cout << "Solution is unstable" << std::endl;
    }
    else if (numUnstableEigenvalues == 0)
    {
      std::cout << "Solution is stable" << std::endl;
    }
    */

    // Store data
    p_data->resize(i+1,noSpikes+4);
    (*p_data)(i,0) = i;
    (*p_data)(i,1) = (*p_parameters)(0);
    (*p_data)(i,2) = arma::norm(*p_solution_new,2);
    (*p_data)(i,3) = numUnstableEigenvalues;
    for (int j=0;j<noSpikes;++j)
    {
      (*p_data)(i,4+j) = (*p_solution_new)(j);
    }

    // Prepare for next step
    (*p_parameters)(0) += ds;
    p_event->SetParameters(0,(*p_parameters)(0));
    p_event->SetParameterStdDev(sigma*(*p_parameters)(0));
    *p_solution_old = *p_solution_new;

    // Save data
    p_data->save("branch.dat",arma::raw_ascii);
  }

  // Clean
  delete p_parameters;
  delete p_data;
  delete p_event;
  delete p_solution_old;
  delete p_solution_new;
  delete p_jacobian;
  delete p_residual_history;
  delete p_newton_solver;
  delete p_eigenvalues;
  delete p_real_eigenvalues;
  delete p_stability;
}
