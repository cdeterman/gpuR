#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <RcppArmadillo.h>

#include "cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
IntegerVector cpp_gpu_two_vec(IntegerVector A_, IntegerVector B_, 
    IntegerVector C_, SEXP sourceCode_, SEXP kernel_function_)
{
    // declarations
    cl_int err;
    std::string sourceCode = as<std::string>(sourceCode_);
    
    std::string kernel_string = as<std::string>(kernel_function_);
    const char* kernel_function = (const char*)kernel_string.c_str();

    // Convert input vectors to cl equivalent
//    const std::vector<int> A = as<std::vector<int> >(A_);
//    const std::vector<int> B = as<std::vector<int> >(B_);
//    std::vector<int> C = as<std::vector<int> >(C_);
    const arma::ivec A = as<arma::ivec >(A_);
    const arma::ivec B = as<arma::ivec >(B_);
    arma::ivec C = as<arma::ivec >(C_);
    
    const int LIST_SIZE = A.size();
    
    // Get available platforms
    std::vector<Platform> platforms;
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

    Context context( CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    if(err != CL_SUCCESS){
        stop("context failed to create"); 
    }
    
    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    if(devices.size() == 0){
        stop("No devices found!");
    }

    // Create a command queue and use the first device
    CommandQueue queue = CommandQueue(context, devices[0], 0, &err);
    
    // Read source file - passed in by R wrapper function            
    Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    // Make program of the source code in the context
    Program program = Program(context, source);
    
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
    Kernel kernel(program, kernel_function, &err);
    if (err != CL_SUCCESS)
    {
       stop("Error: Failed to create compute kernel!\n");
    }
    
//        std::cout << "Kernel made" << std::endl;

    // Create memory buffers
    Buffer bufferA = Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &err);
    Buffer bufferB = Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &err);
    Buffer bufferC = Buffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int), NULL, &err);

//        std::cout << "memory mapped" << std::endl;

    // Copy lists A and B to the memory buffers
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), &A[0]);
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, LIST_SIZE * sizeof(int), &B[0]);

    // Set arguments to kernel
    err = kernel.setArg(0, bufferA);
    err = kernel.setArg(1, bufferB);
    err = kernel.setArg(2, bufferC);
    
    // Run the kernel on specific ND range
    NDRange global(LIST_SIZE);
    NDRange local(1);
    err = queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
    
    // Read buffer C into a local list        
    err = queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, LIST_SIZE * sizeof(cl_int), &C[0]);
    
    return wrap(C);
}
