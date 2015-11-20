
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "gpuR/eigen_helpers.hpp"
#include "gpuR/cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
void cpp_gpuMatrix_igemm(SEXP ptrA_, SEXP ptrB_, SEXP ptrC_,
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
//    const char* kernel_function = (const char*)kernel_string.c_str();

    Rcpp::XPtr<dynEigen<int> > ptrA(ptrA_);
    Rcpp::XPtr<dynEigen<int> > ptrB(ptrB_);
    Rcpp::XPtr<dynEigen<int> > ptrC(ptrC_);
    
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > Am(ptrA->ptr(), ptrA->nrow(), ptrA->ncol());
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > Bm(ptrB->ptr(), ptrB->nrow(), ptrB->ncol());
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > Cm(ptrC->ptr(), ptrC->nrow(), ptrC->ncol());
    
    int Mdim = Am.cols();
    int Ndim = Bm.rows();
    int Pdim = Am.rows();
//    int Kdim = Bm.cols();
    
    const int szA = Am.size();
    const int szB = Bm.size();
    const int szC = Cm.size();
    
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
        
//        std::cout << "Context Made" << std::endl;

    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    if(devices.size() < 1){
        stop("No GPU devices found");
    }
        
//        std::cout << "Devices Found" << std::endl;

        
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
        
//        std::cout << "Program Built" << std::endl;
        
    // Make kernel
    Kernel kernel(program, "iMatMult", &err);
//        Kernel kernel(program, kernel_function, &err);
    
    if (err != CL_SUCCESS)
    {
       stop("Error: Failed to create compute kernel!\n");
    }
        
        
//        cl_int wgSize = kernel.getWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
//        std::cout << "Kernel made" << std::endl;

    // Create memory buffers
    Buffer bufferA = Buffer(context, CL_MEM_READ_ONLY, szA * sizeof(int), NULL, &err);
    Buffer bufferB = Buffer(context, CL_MEM_READ_ONLY, szB * sizeof(int), NULL, &err);
    Buffer bufferC = Buffer(context, CL_MEM_WRITE_ONLY, szC * sizeof(int), NULL, &err);

    // Copy lists A and B to the memory buffers
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, szA * sizeof(int), &Am(0));
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, szB * sizeof(int), &Bm(0));

        // Set arguments to kernel
//        NDRange localWorkSize = NDRange(16, 16);
//        NDRange globalWorkSize = NDRange(1024, 1024);
        
//        err = kernel.setArg(4, sizeof(int), &Mdim);

    err = kernel.setArg(0, sizeof(int), &Mdim);
    err = kernel.setArg(1, sizeof(int), &Ndim);
    err = kernel.setArg(2, sizeof(int), &Pdim);

    err = kernel.setArg(3, bufferA);
    err = kernel.setArg(4, bufferB);
    err = kernel.setArg(5, bufferC);
//        err = kernel.setArg(4, 16*sizeof(int), NULL);
        
//        err = kernel.setArg(3, sizeof(int), &Mdim);
//        err = kernel.setArg(4, sizeof(int), &Ndim);
        
//        localWorkSize[0] = 16;
//        localWorkSize[1] = 16;
//        globalWorkSize[0] = 1024;
//        globalWorkSize[1] = 1024;
        
    // Run the kernel on specific ND range
//        err = queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
    err = queue.enqueueNDRangeKernel(kernel, NullRange, 
                                    NDRange(Mdim, Ndim), NullRange);

//        err = queue.enqueueNDRangeKernel(kernel, NullRange, global, NullRange);
        
    // Read buffer C into a local list        
    err = queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, 
                                szC * sizeof(int), &Cm(0));
}
