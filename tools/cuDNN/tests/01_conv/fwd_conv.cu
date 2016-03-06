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

  int batchsize, featuremaps, h, w;
  batchsize = featuremaps = 1;
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

  network_t nets;

  // conv forward
  // input, output, kernel_dim
  Layer_t conv1(1,20,5);



  nets.FwdConv(conv1, batchsize, featuremaps, h, w, srcData_d, &dstData_d);

  /*
  mnist.poolForward(n, c, h, w, dstData_d, &srcData_d);

  //printf("=> pool 1 -> before fcf : %d, %d, %d, %d \n", n, c,h,w);

  Layer_t ip1(c*h*w,500,1);
  mnist.fullyConnectedForward(ip1, n, c, h, w, srcData_d, &dstData_d);

  //printf("=> before act : %d, %d, %d, %d \n", n, c,h,w);
  //mnist.activationForward(n, c, h, w, dstData_d, &srcData_d);

  printf("=> before softmax : %d, %d, %d, %d \n", n, c,h,w);
  mnist.softmaxForward(n, c, h, w, dstData_d, &srcData_d);
*/

  delete [] imgData_h; 

  checkCudaErrors( cudaFree(srcData_d) );
  checkCudaErrors( cudaFree(dstData_d) );


  return (EXIT_SUCCESS);
}
