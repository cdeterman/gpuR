
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
void cpp_gpuMatrix_iaxpy(SEXP alpha_, SEXP ptrA_, SEXP ptrB_,
    SEXP sourceCode_)
{
    // declarations
    cl_int err = 0;
    std::string sourceCode = as<std::string>(sourceCode_);
    
    #if defined(__APPLE__) || defined(__MACOSX)
        #ifdef HAVE_OPENCL_CL2_HPP
            std::vector<std::string> sourceCodeVec;
            sourceCodeVec.push_back(sourceCode);
        #endif
    #else
        #ifdef HAVE_CL_CL2_HPP
            std::vector<std::string> sourceCodeVec;
            sourceCodeVec.push_back(sourceCode);
        #endif
    #endif
    
//    std::string kernel_string = as<std::string>(kernel_function_);
//    const char* kernel_function = kernel_string.data();
    
//    XPtr<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > ptrA(ptrA_);
//    XPtr<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > ptrB(ptrB_);
    
//    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->data(), ptrA->rows(), ptrA->cols());
//    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->data(), ptrB->rows(), ptrB->cols());
    
    XPtr<dynEigenMat<int> > ptrA(ptrA_);
    XPtr<dynEigenMat<int> > ptrB(ptrB_);
    
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > refA = ptrA->data();
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > refB = ptrB->data();
    
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > Am(refA.data(), ptrA->nrow(), ptrA->ncol());
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > Bm(refB.data(), ptrB->nrow(), ptrB->ncol());
    
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
    #if defined(__APPLE__) || defined(__MACOSX)
        #ifndef HAVE_OPENCL_CL2_HPP
            int pl;
            std::pair <const char*, int> sourcePair;
            sourcePair = std::make_pair(sourceCode.c_str(), pl);
            Program::Sources source(1, sourcePair);
        #else
            Program::Sources source(sourceCodeVec);
        #endif
    #else
        #ifndef HAVE_CL_CL2_HPP
            int pl;
            std::pair <const char*, int> sourcePair;
            sourcePair = std::make_pair(sourceCode.c_str(), pl);
            Program::Sources source(1, sourcePair);
        #else
            Program::Sources source(sourceCodeVec);
        #endif
    #endif

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
