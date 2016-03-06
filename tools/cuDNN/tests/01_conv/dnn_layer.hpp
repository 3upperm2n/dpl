#ifndef DNN_LAYER_H
#define DNN_LAYER_H

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "cuda_checking.hpp"

// todo: inheritance for different layer types 
//
//
class Layer_t
{
  public:
	int inputs;
	int outputs;
	// linear dimension (i.e. size is kernel_dim * kernel_dim)
	// todo: more configuration 
	int kernel_dim;

	float *wei_h;
	float *wei_d;

	float *bias_h;
	float *bias_d;

	size_t wei_len;
	size_t bias_len;

	Layer_t(): inputs(0), outputs(0), kernel_dim(0), wei_h(NULL), wei_d(NULL), 
  bias_h(NULL), bias_d(NULL) {};

	Layer_t(int _inputs, int _outputs, int _kernel_dim) : 
	  inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
  {
	// weights
	wei_len = inputs * outputs * kernel_dim * kernel_dim;
	wei_h = new float[wei_len];
#pragma unroll
	for(int i=0; i<wei_len; i++) {
	  wei_h[i] = 1.f;	
	}
	checkCudaErrors( cudaMalloc(&wei_d, wei_len * sizeof(float)) );
	checkCudaErrors( cudaMemcpy(wei_d, wei_h, 
		  wei_len * sizeof(float), cudaMemcpyHostToDevice) );

	// bias
	bias_len = outputs;
	bias_h = new float[bias_len];
#pragma unroll
	for(int i=0; i<bias_len; i++) {
	  bias_h[i] = 1.f;	
	}
	checkCudaErrors( cudaMalloc(&bias_d, bias_len * sizeof(float)) );
	checkCudaErrors( cudaMemcpy(bias_d, bias_h, 
		  bias_len * sizeof(float), cudaMemcpyHostToDevice) );

  }

	~Layer_t()
	{
	  delete [] wei_h;
	  delete [] bias_h;
	  checkCudaErrors( cudaFree(wei_d) );
	  checkCudaErrors( cudaFree(bias_d) );
	}
};



#endif
