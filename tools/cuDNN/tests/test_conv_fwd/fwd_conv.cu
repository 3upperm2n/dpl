#include <sstream>
#include <fstream>
#include <iostream>

#include <cublas_v2.h>
#include <cudnn.h>

#include "argopt.hpp"
#include "cuda_checking.hpp"
#include "dnn_network.hpp"
#include "dnn_layer.hpp"



int main(int argc, char **argv) {
  ARG arg;
  arg.parsing(argc, argv);

  int n, c, h, w;
  n = c = 1;
  w = arg.image.width;
  h = arg.image.height;


  float *imgData_h = new float[w * h];

  for (int i = 0; i < h; i++)                                       
  {                                                                       
	for (int j = 0; j < w; j++)                                   
	{                                                                   
	  int idx = w*i + j;                                        
	  // 0.0 - 1.0
	  imgData_h[idx] = (float)(rand())/ (float)(RAND_MAX);
	}                                                                   
  }                                                                       

  float *srcData_d;
  float *dstData_d = NULL;

  cudaMalloc(&srcData_d, w*h*sizeof(float));
  cudaMemcpy(srcData_d, imgData_h, h*w*sizeof(float), cudaMemcpyHostToDevice);

  network_t mnist;

  // conv forward
  // input, output, kernel_dim
  Layer_t conv1(1,20,5);

  mnist.convForward(conv1, n, c, h, w, srcData_d, &dstData_d);

  delete [] imgData_h; 

  checkCudaErrors( cudaFree(srcData_d) );
  checkCudaErrors( cudaFree(dstData_d) );


  return (EXIT_SUCCESS);
}
