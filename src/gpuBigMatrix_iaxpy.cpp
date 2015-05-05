#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

//#include <CL/cl.h>

#include <RcppArmadillo.h>

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

#include "arma_helpers.hpp"
#include "cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
void cpp_gpuBigMatrix_iaxpy(SEXP alpha_, SEXP A_, SEXP B_,
    SEXP sourceCode_, SEXP kernel_function_)
{
    // declarations
    cl_int err;
    std::string sourceCode = as<std::string>(sourceCode_);
    
    std::string kernel_string = as<std::string>(kernel_function_);
    const char* kernel_function = kernel_string.data();
                              
    const arma::Mat<int> Am = ConvertBMtoArma<int>(A_);
    arma::Mat<int> Bm = ConvertBMtoArma<int>(B_);
    
    const int N = Am.n_elem;
    const int alpha = as<int>(alpha_);

//    Am.print("A Matrix");
//    Bm.print("B Matrix");
    
    // Get available platforms
    std::vector<Platform> platforms;
    Platform::get(&platforms);        
    
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
            //std::cerr << log << std::endl;
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
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, N * sizeof(int), &Am[0]);
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, N * sizeof(int), &Bm[0]);

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
    err = queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, N * sizeof(int), &Bm[0]);
}
