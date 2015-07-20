#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <CL/cl.hpp>

#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"
#include "gpuR/cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
void cpp_gpuMatrix_iaxpy(SEXP alpha_, SEXP ptrA_, SEXP ptrB_,
    SEXP sourceCode_)
{
    // declarations
    cl_int err;
    std::string sourceCode = as<std::string>(sourceCode_);
    
//    std::string kernel_string = as<std::string>(kernel_function_);
//    const char* kernel_function = kernel_string.data();
        
    Rcpp::XPtr<dynEigen<int> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<int> > ptrB(ptrB_);
    
    MapMat<int> Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    MapMat<int> Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    
    const int N = Am.size();
    const int alpha = as<int>(alpha_);
    
    // Get available platforms
    std::vector<Platform> platforms;
    getPlatforms(platforms); // cl_helpers.hpp       
    
    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(),
        0
    };

    Context context = createContext(CL_DEVICE_TYPE_GPU, cps, err);
        
    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    if(devices.size() < 1){
        stop("No GPU devices found");
    }
        
    // Create a command queue and use the first device
    CommandQueue queue = CommandQueue(context, devices[0], 0, &err);

    // Read source file - passed in by R wrapper function
    int pl;
    Program::Sources source(1, 
        std::make_pair(sourceCode.c_str(), pl));
       
    // Make program of the source code in the context
    Program program = Program(context, source);
    if (err != CL_SUCCESS)
    {
       stop("Error: Failed to create compute program!\n");
    }

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
        }
        stop("program failed to build");
    }
        
        
    // Make kernel
    Kernel kernel(program, "iaxpy", &err);
//        Kernel kernel(program, kernel_function, &err);
    
    if (err != CL_SUCCESS)
    {
       stop("Error: Failed to create compute kernel!\n");
    }
        
        
//        cl_int wgSize = kernel.getWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
//        std::cout << "Kernel made" << std::endl;

    // Create memory buffers
    Buffer bufferA = Buffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);
    Buffer bufferB = Buffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &err);

    // Copy lists A and B to the memory buffers
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, N * sizeof(int), &Am(0));
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, N * sizeof(int), &Bm(0));

    // Set arguments to kernel
    err = kernel.setArg(0, alpha);
    err = kernel.setArg(1, bufferA);
    err = kernel.setArg(2, bufferB);
        
    // Run the kernel on specific ND range
//    err = queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
    err = queue.enqueueNDRangeKernel(kernel, NullRange, 
                                    NDRange(N), NullRange);

//        err = queue.enqueueNDRangeKernel(kernel, NullRange, global, NullRange);
        
    // Read buffer C into a local list        
    err = queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, N * sizeof(int), &Bm(0));
}
