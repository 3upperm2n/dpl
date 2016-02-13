/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "ImageIO.h"

#define value_type float

#define IMAGE_H 28
#define IMAGE_W 28

const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";

const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";
const char *ip1_bin = "ip1.bin";
const char *ip1_bias_bin = "ip1.bias.bin";
const char *ip2_bin = "ip2.bin";
const char *ip2_bias_bin = "ip2.bias.bin";

/********************************************************
 * Prints the error message, and exits
 * ******************************************************/

#define EXIT_WAIVED 0

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
}

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("data/") + std::string(fname));
}

struct Layer_t
{
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0){};
    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readBinaryFile(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
        readBinaryFile(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }
    ~Layer_t()
    {
        delete [] data_h;
        checkCudaErrors( cudaFree(data_d) );
    }
private:
    void readBinaryFile(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
        std::stringstream error_s;
        if (!dataFile)
        {
            error_s << "Error opening file " << fname; 
            FatalError(error_s.str());
        }
        int size_b = size*sizeof(value_type);
        *data_h = new value_type[size];
        if (!dataFile.read ((char*) *data_h, size_b)) 
        {
            error_s << "Error reading file " << fname; 
            FatalError(error_s.str());
        }
        checkCudaErrors( cudaMalloc(data_d, size_b) );
        checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                    size_b,
                                    cudaMemcpyHostToDevice) );
    }
};

void printDeviceVector(int size, value_type* vec_d)
{
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

class network_t
{
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cublasHandle_t cublasHandle;
    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );

        checkCudaErrors( cublasCreate(&cublasHandle) );
    }
    void destroyHandles()
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
  public:
    network_t()
    {
        dataType = CUDNN_DATA_FLOAT;
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
    };
    ~network_t()
    {
        destroyHandles();
    }
    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
    }
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
    void convoluteForward(const Layer_t& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;

        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h, w) );

        checkCUDNN( cudnnSetFilter4dDescriptor(filterDesc,
                                              dataType,
                                              conv.outputs,
                                              conv.inputs, 
                                              conv.kernel_dim,
                                              conv.kernel_dim) );
 
        checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,
                                                   // srcTensorDesc,
                                                    //filterDesc,
                                                    0,0, // padding
                                                    1,1, // stride
                                                    1,1, // upscale
                                                    CUDNN_CROSS_CORRELATION) );
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                &n, &c, &h, &w) );

        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                0,
                                                &algo
                                                ) );
        resize(n*c*h*w, dstData);
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
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

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
};

inline bool IsAppBuiltAs64()
{
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined(__aarch64__)
    return 1;
#else
    return 0;
#endif
}

int main(int argc, char *argv[])
{
    if (argc > 2)
    {
        std::cout << "Test usage:\nmnistCUDNN [image]\nExiting...\n";
        exit(EXIT_FAILURE);
    }
    
    if(sizeof(void*) != 8)
    {
#ifndef __aarch32__
      std::cout <<"With the exception of ARM, " << argv[0] << " is only supported on 64-bit OS and the application must be built as a 64-bit target. Test is being waived.\n";
      exit(EXIT_WAIVED);
#endif
    }

    std::string image_path;
    network_t mnist;

    Layer_t conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
    Layer_t conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
    Layer_t   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
    Layer_t   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);
    
    if (argc == 1)
    {
        int i1,i2,i3;
        get_path(image_path, first_image, argv[0]);
        i1 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
        
        get_path(image_path, second_image, argv[0]);
        i2 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
        
        get_path(image_path, third_image, argv[0]);
        i3 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);

        std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
        if (i1 != 1 || i2 != 3 || i3 != 5)
        {
            std::cout << "\nTest failed!\n";
            FatalError("Prediction mismatch");
        }
        else
        {
            std::cout << "\nTest passed!\n";
        }
    }
    else
    {
        int i1 = mnist.classify_example(argv[1], conv1, conv2, ip1, ip2);
        std::cout << "\nResult of classification: " << i1 << std::endl;
    }
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
