
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"
#include "gpuR/cl_helpers.hpp"

//using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
void cpp_gpu_two_vec(SEXP ptrA_, SEXP ptrB_, 
    SEXP ptrC_, SEXP sourceCode_, SEXP kernel_function_)
{
    // declarations
    cl_int err = 0;
    std::string sourceCode = as<std::string>(sourceCode_);
    
    #ifdef HAVE_CL_CL2_HPP
        std::vector<std::string> sourceCodeVec;
        sourceCodeVec.push_back(sourceCode);
    #endif
    
    std::string kernel_string = as<std::string>(kernel_function_);
    const char* kernel_function = (const char*)kernel_string.c_str();

    // Convert input vectors to cl equivalent
//    const std::vector<int> A = as<std::vector<int> >(A_);
//    const std::vector<int> B = as<std::vector<int> >(B_);
//    std::vector<int> C = as<std::vector<int> >(C_);

    Rcpp::XPtr<dynEigenVec<int> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigenVec<int> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigenVec<int> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 1> > A(ptrA->ptr(), ptrA->length());
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 1> > B(ptrB->ptr(), ptrB->length());
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 1> > C(ptrC->ptr(), ptrC->length());

    const int LIST_SIZE = A.size();
    
    // Get available platforms
    std::vector<cl::Platform> platforms;
    getPlatforms(platforms); // cl_helpers.hpp
    
    if(platforms.size() == 0){
        stop("No platforms found. Check OpenCL installation!");
    }

    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(),
        0
    };

    cl::Context context( CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    if(err != CL_SUCCESS){
        stop("context failed to create"); 
    }
    
    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    if(devices.size() == 0){
        stop("No devices found!");
    }

    // Create a command queue and use the first device
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0], 0, &err);
    
    /// Read source file - passed in by R wrapper function
    #ifndef HAVE_CL_CL2_HPP
        int pl;
        std::pair <const char*, int> sourcePair;
        sourcePair = std::make_pair(sourceCode.c_str(), pl);
        cl::Program::Sources source(1, sourcePair);
    #else
        cl::Program::Sources source(sourceCodeVec);
    #endif
        
    // Make program of the source code in the context
    cl::Program program = cl::Program(context, source);
    
    // Build program for these specific devices
    try
    {
        program.build(devices);
    }
    catch (cl::Error error)
    {
        if (error.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            // Get the build log for the first device
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            stop(log);
            //std::cerr << log << std::endl;
        }
        stop("program failed to build");
    }
    
    // Make kernel
//        Kernel kernel(program, "vector_add", &err);
    cl::Kernel kernel(program, kernel_function, &err);
    if (err != CL_SUCCESS)
    {
       stop("Error: Failed to create compute kernel!\n");
    }
    
//        std::cout << "Kernel made" << std::endl;

    // Create memory buffers
    cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &err);
    cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &err);
    cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int), NULL, &err);

//        std::cout << "memory mapped" << std::endl;

    // Copy lists A and B to the memory buffers
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), &A(0));
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, LIST_SIZE * sizeof(int), &B(0));

    // Set arguments to kernel
    err = kernel.setArg(0, bufferA);
    err = kernel.setArg(1, bufferB);
    err = kernel.setArg(2, bufferC);
    
    // Run the kernel on specific ND range
    cl::NDRange global_range(LIST_SIZE);
    cl::NDRange local_range(1);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range);
    
    // Read buffer C into a local list        
    err = queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, LIST_SIZE * sizeof(int), &C(0));
//    C.print("C vector!");
//    return wrap(C);
}
