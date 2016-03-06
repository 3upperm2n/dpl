#ifndef DNN_NETWORK_H
#define DNN_NETWORK_H

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "cuda_checking.hpp"

#include "dnn_layer.hpp"



class network_t
{
  public:
    cudnnDataType_t dataType;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    cudnnTensorFormat_t tensorFormat;
	// tensors
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
	// filter
    cudnnFilterDescriptor_t filterDesc;
	// conv
    cudnnConvolutionDescriptor_t convDesc;
	// pooling
    cudnnPoolingDescriptor_t poolingDesc;

    void createHandles();
    void destroyHandles();

    network_t()
    {
        dataType = CUDNN_DATA_FLOAT;		  // float
        tensorFormat = CUDNN_TENSOR_NCHW;	  // row major for tensor
        createHandles();    
    }

    ~network_t()
    {
        destroyHandles();
    }

    void resize(int size, float **data_d)
    {
        if (*data_d != NULL)
            checkCudaErrors( cudaFree(*data_d) );
        checkCudaErrors( cudaMalloc(data_d, size*sizeof(float)) );
    }

	void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, 
		const Layer_t& layer, int c, float *data);

	//-----------------------------------------------------------------------//
	// convolution using cudnn 
	//-----------------------------------------------------------------------//
	void FwdConv(const Layer_t& conv,
			  int& n,	// batch size 
			  int& c,	// feature maps
			  int& h,	// height of image
			  int& w,	// width of image
			  float *srcData, 
			  float **dstData);

	void poolForward( int& n, int& c, int& h, int& w, 
			  float *srcData, float ** dstData);

	void fullyConnectedForward(const Layer_t& ip,
		int& n, int& c, int& h, int& w, float* srcData, float** dstData);


    void activationForward(int n, int c, int h, int w, float* srcData, float ** dstData);

    void softmaxForward(int n, int c, int h, int w, float* srcData, float** dstData);

};

void network_t::createHandles()
{
  checkCUDNN( cudnnCreate(&cudnnHandle) );
  checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
  checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
  checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
  checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
  checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
  // convolution
  checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );

  checkCudaErrors( cublasCreate(&cublasHandle) );
} 


void network_t::destroyHandles()
{
  checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
  checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
  checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
  checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
  checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
  checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
  checkCUDNN( cudnnDestroy(cudnnHandle) );

  checkCudaErrors( cublasDestroy(cublasHandle) );
}


//---------------------------------------------------------------------------//
// convolution using cudnn
//---------------------------------------------------------------------------//
void network_t::FwdConv(const Layer_t& conv,
	int& n,	// batch size 
	int& c, // feature maps
	int& h, // height of image
	int& w, // width of image
	float *srcData, 
	float **dstData)
{
  printf("=> input %d\n", conv.inputs);
  printf("=> output %d\n", conv.outputs);
  printf("=> kernel dim : %d\n", conv.kernel_dim);
  printf("=> src : %d %d %d %d\n", n, c, h, w);


  cudnnConvolutionFwdAlgo_t algo;

  // prepare src
  checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
		tensorFormat,
		dataType,
		n, c, h, w) );

  checkCUDNN( cudnnSetFilter4dDescriptor(filterDesc,
		dataType,
		conv.outputs,		  // output feature maps
		conv.inputs,		  // input feature maps 
		conv.kernel_dim,	  // height of input filter
		conv.kernel_dim) );	  // width of input filter


  checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,
		0,0, // padding
		1,1, // stride
		1,1, // upscale
		CUDNN_CROSS_CORRELATION) );

  // find dimension of convolution output
  checkCUDNN( cudnnGetConvolution2dForwardOutputDim(convDesc,
		srcTensorDesc,
		filterDesc,
		&n, &c, &h, &w) );

  // prepare dest
  checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h, w) );

  // update the convolution algorithm
  checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		srcTensorDesc,
		filterDesc,
		convDesc,
		dstTensorDesc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&algo
		) );

  // resize output data 
//  printf("=> dst: %d %d %d %d\n", n, c, h, w);
  resize(n*c*h*w, dstData);
//  printf("resize : %d\n", n * c * h * w);

  size_t sizeInBytes=0;
  void* workSpace=NULL;
  checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		srcTensorDesc,
		filterDesc,
		convDesc,
		dstTensorDesc,
		algo,
		&sizeInBytes) );

  if (sizeInBytes!=0)
  {
	checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
  }

  //printf("=> sizeInBytes : %lu \n", sizeInBytes);

  float alpha = 1.f;
  float beta  = 0.f;

  checkCUDNN( cudnnConvolutionForward(cudnnHandle,
		&alpha,				// alpha
		srcTensorDesc,		// src tensor descriptor
		srcData,			// src data
		filterDesc,			// filter descriptor
		conv.wei_d,		// conv data : weight tensor
		convDesc,			// conv descriptor
		algo,				// conv algo
		workSpace,			// workspace
		sizeInBytes,		// bytes in workspaces
		&beta,				// 
		dstTensorDesc,		// dest tensor descriptor
		*dstData) );		// dest tensor

  // add bias
  //addBias(dstTensorDesc, conv, c, *dstData);

  if (sizeInBytes!=0)
  {
	checkCudaErrors( cudaFree(workSpace) );
  }
}

void network_t::addBias(const cudnnTensorDescriptor_t& dstTensorDesc, 
	const Layer_t& layer, int c, float *data)
{
  checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
		tensorFormat,
		dataType,
		1, c,
		1,
		1) );

  float alpha = float(1);
  float beta  = float(1);
  checkCUDNN( cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C,
		&alpha, biasTensorDesc,
		layer.bias_d,
		&beta,
		dstTensorDesc,
		data) );
}

void network_t::poolForward( int& n, int& c, int& h, int& w,
	float *srcData, float ** dstData)
{
  checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc,
		CUDNN_POOLING_MAX,
		2, 2, // window
		0, 0, // padding
		2, 2  // stride
		) );

  checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w ) );
  h = h / 2; w = w / 2;
  checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w) );

  resize(n*c*h*w, dstData);

  float alpha = float(1);
  float beta  = float(0);
  checkCUDNN( cudnnPoolingForward(cudnnHandle,
		poolingDesc,
		&alpha,
		srcTensorDesc,
		srcData,
		&beta,
		dstTensorDesc,
		*dstData) );
}

void network_t::fullyConnectedForward(const Layer_t& ip,
	int& n, int& c, int& h, int& w, float* srcData, float** dstData)
{
  if (n != 1)
  {
	FatalError("Not Implemented"); 
  }
  int dim_x = c*h*w;
  int dim_y = ip.outputs;
  resize(dim_y, dstData);

  float alpha = float(1), beta = float(1);

  // place bias into dstData
  checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, 
		dim_y*sizeof(float), cudaMemcpyDeviceToDevice) );

  checkCudaErrors( cublasSgemv(cublasHandle, CUBLAS_OP_T,
		dim_x, dim_y,
		&alpha,
		ip.wei_d, dim_x,
		srcData, 1,
		&beta,
		*dstData, 1) );

  h = 1; w = 1; c = dim_y;
}

void network_t::activationForward(int n, int c, int h, int w, float *srcData, float** dstData)
{
  resize(n*c*h*w, dstData);
  checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w) );
  checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w) );
  float alpha = float(1);
  float beta  = float(0);
  checkCUDNN( cudnnActivationForward(cudnnHandle,
		CUDNN_ACTIVATION_RELU,
		&alpha,
		srcTensorDesc,
		srcData,
		&beta,
		dstTensorDesc,
		*dstData) );    
}

void network_t::softmaxForward(int n, int c, int h, int w, float* srcData, float** dstData)
{
  resize(n*c*h*w, dstData);

  checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w) );
  checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w) );

  float alpha = float(1);
  float beta  = float(0);

  checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
		CUDNN_SOFTMAX_ACCURATE ,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha,
		srcTensorDesc,
		srcData,
		&beta,
		dstTensorDesc,
		*dstData) );
}
#endif
