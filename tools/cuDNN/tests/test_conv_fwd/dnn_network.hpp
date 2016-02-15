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
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cublasHandle_t cublasHandle;

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

	void convForward(const Layer_t& conv,
			  int& n,	// batch size 
			  int& c,	// feature maps
			  int& h,	// height of image
			  int& w,	// width of image
			  float *srcData, 
			  float ** stData);

	/*
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t& layer, int c, value_type *data)
    {
        checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                1, c,
                                                1,
                                                1) );
        value_type alpha = value_type(1);
        value_type beta  = value_type(1);
        checkCUDNN( cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C,
                                      &alpha, biasTensorDesc,
                                      layer.bias_d,
                                      &beta,
                                      dstTensorDesc,
                                      data) );
    }
    void fullyConnectedForward(const Layer_t& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        resize(dim_y, dstData);

        value_type alpha = value_type(1), beta = value_type(1);
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        checkCudaErrors( cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                      dim_x, dim_y,
                                      &alpha,
                                      ip.data_d, dim_x,
                                      srcData, 1,
                                      &beta,
                                      *dstData, 1) );

        h = 1; w = 1; c = dim_y;
    }
  */

/*
    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData)
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
        value_type alpha = value_type(1);
        value_type beta = value_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
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
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
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
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
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
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            CUDNN_ACTIVATION_RELU,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }

    int classify_example(const char* fname, const Layer_t& conv1,
                          const Layer_t& conv2,
                          const Layer_t& ip1,
                          const Layer_t& ip2)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        std::string sFilename(fname);
        std::cout << "Loading image " << sFilename << std::endl;
        // load gray-scale image from disk
        try
        {
            npp::loadImage(sFilename, oHostSrc);
        }
        catch (npp::Exception &rException)
        {
            FatalError(rException.toString());
        }
        // Plot to console and normalize image to be in range [0,1]
        for (int i = 0; i < IMAGE_H; i++)
        {
            for (int j = 0; j < IMAGE_W; j++)
            {   
                int idx = IMAGE_W*i + j;
                imgData_h[idx] = *(oHostSrc.data() + idx) / value_type(255);
            }
        }

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );

        n = c = 1; h = IMAGE_H; w = IMAGE_W;


        convoluteForward(conv1, n, c, h, w, srcData, &dstData);




        poolForward(n, c, h, w, dstData, &srcData);

        convoluteForward(conv2, n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);

        fullyConnectedForward(ip1, n, c, h, w, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);

        fullyConnectedForward(ip2, n, c, h, w, srcData, &dstData);
        softmaxForward(n, c, h, w, dstData, &srcData);

        const int max_digits = 10;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, srcData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        int id = 0;
        for (int i = 1; i < max_digits; i++)
        {
            if (result[id] < result[i]) id = i;
        }

        std::cout << "Resulting weights from Softmax:" << std::endl;
        printDeviceVector(n*c*h*w, srcData);

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }
	*/
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

void network_t::convForward(const Layer_t& conv,
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
  printf("=> dst: %d %d %d %d\n", n, c, h, w);
  resize(n*c*h*w, dstData);
  printf("resize : %d\n", n * c * h * w);

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

  printf("=> sizeInBytes : %lu \n", sizeInBytes);

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
  /*
	 addBias(dstTensorDesc, conv, c, *dstData);
	 if (sizeInBytes!=0)
	 {
	 checkCudaErrors( cudaFree(workSpace) );
	 }
	 */
}


#endif
