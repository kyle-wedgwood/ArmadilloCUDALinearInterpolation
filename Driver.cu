#include <iostream>
#include <iomanip>
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
  EventDrivenMap* p_problem = new EventDrivenMap(p_parameters,noReal);
  //p_problem->SetParameterStdDev(0.5);

  // Switch on debugging
  p_problem->SetDebugFlag(1);

  // Initial guess
  arma::vec u0 = arma::vec(noSpikes);
  u0 << 0.3310f << 0.6914f << 1.3557f;
  arma::vec u1(u0);

  // For computing Jacobian via finite differences
  arma::vec f0 = arma::vec(noSpikes);
  arma::vec f1 = arma::vec(noSpikes);
  arma::mat jac = arma::mat(noSpikes,noSpikes);

  // Vector of epsilons
  unsigned int N_steps = 10;

  double eps_max = log10(1.0e-2);
  double eps_min = log10(1.0e-5);
  double deps = (eps_max-eps_min)/(N_steps-1);
  double epsilon = eps_min;
  double matrix_action_norm;

  // Test vector
  arma::vec test_vec = arma::vec(noSpikes,arma::fill::randn);
  test_vec = arma::normalise(test_vec);

  arma::vec Jv = arma::vec(noSpikes);

  // File to save data
  std::ofstream file;
  file.open("MatrixAction.dat");
  file << "EPS" << "\t" << "JV" << "\r\n";
  std::cout << std::setw(20)
            << std::left
            << "EPS"
            << std::setw(20)
            << std::left
            << "JV"
            << std::endl;

  // Calculate initial f
  p_problem->ComputeF(u0,f0);
  printf("Zero'th fn.\n");
  getchar();

  // Now loop over steps
  for (int i=0;i<N_steps;++i)
  {
    epsilon = pow(10,epsilon);
    u1 = u0;

    // Construct Jacobian
    for (int j=0;j<noSpikes;++j)
    {
      if (j>0)
      {
        u1(j-1) = u0(j-1);
      }
      u1(j) += epsilon;
      p_problem->ComputeF(u1,f1);
      printf("%d'th fn.\n",j+1);
      getchar();

      jac.col(j) = (f1-f0)*pow(epsilon,-1);
    }
    // Restore final element of u
    u1(noSpikes-1) = u0(noSpikes-1);

    // Calculate Jacobian action
    Jv = jac*test_vec;
    //std::cout << Jv << std::endl;
    matrix_action_norm = arma::norm(Jv,2);

    // Save and display data
    file << epsilon << "\t" << matrix_action_norm << "\r\n";
    std::cout << std::setprecision(7)
              << std::setw(20)
              << epsilon
              << std::setprecision(10)
              << std::setw(20)
              << matrix_action_norm << std::endl;

    // Prepare for next step
    epsilon = log10(epsilon);
    epsilon += deps;
  }

  // Clean
  file.close();
  delete p_parameters;
  delete p_problem;
}
