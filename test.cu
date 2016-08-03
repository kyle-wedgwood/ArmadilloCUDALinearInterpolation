#include <iostream>
#include <cstdlib>
#include <armadillo>
#define npts 1000000
#define noThreads 512

using namespace std;

__global__ void testKernel(float *a,float *b)
{
  int i = threadIdx.x+blockDim.x*blockIdx.x;
  if (i<npts)
  {
    a[i] = 2*b[i];
  }
}

int main()
{
  arma::fvec* host_dummy  = new arma::fvec(npts);
  arma::fvec* host_dummy2 = new arma::fvec(npts);

  float *dev_dummy;
  cudaMalloc(&dev_dummy,npts*sizeof(float));
  float *dev_dummy2;
  cudaMalloc(&dev_dummy2,npts*sizeof(float));

  for (int i=0;i<npts;++i)
  {
    (*host_dummy)[i] = i/2.0;
  }

  float *dummyPtr = host_dummy->begin();
  //for (int i=0;i<npts;++i)
  //{
  //  printf("%f\n",*(dummyPtr+i));
  //}
  //getchar();
  cudaMemcpy(dev_dummy,host_dummy->begin(),npts*sizeof(float),cudaMemcpyHostToDevice);

  testKernel<<<(npts+noThreads-1)/noThreads,noThreads>>>(dev_dummy2,dev_dummy);

  cudaMemcpy(host_dummy2->begin(),dev_dummy2,npts*sizeof(float),cudaMemcpyDeviceToHost);

  //cout << *host_dummy2 << endl;

  arma::vec realPart(10,arma::fill::randu);
  arma::vec imagPart(10,arma::fill::randu);

  arma::cx_vec randomComplexVector = arma::cx_vec(realPart,imagPart);

  cout << abs(realPart) << endl;
  cout << abs(imagPart) << endl;
  cout << abs(randomComplexVector) << endl;
  cout << sqrt(pow(realPart,2)+pow(imagPart,2)) << endl;

  // Now testing conversion to float
  arma::fvec* realPartFloat = new arma::fvec(10);
  *realPartFloat = arma::conv_to<arma::fvec>::from(realPart);

  cout << *realPartFloat << endl;

  return 0;
}
