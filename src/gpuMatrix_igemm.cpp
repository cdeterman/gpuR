
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;


//[[Rcpp::export]]
void cpp_gpuMatrix_igemm(SEXP ptrA_, SEXP ptrB_, SEXP ptrC_,
    SEXP sourceCode_, int device_type)
{
    // declarations
    cl_int err = 0;
    std::string sourceCode = as<std::string>(sourceCode_);
    cl_device_type ocl_device;
    
    Program::Sources sources;
    sources.push_back({sourceCode.c_str(), sourceCode.length()});
    
//    #if defined(__APPLE__) || defined(__MACOSX)
//        #ifdef HAVE_OPENCL_CL2_HPP
//            std::vector<std::string> sourceCodeVec;
//            sourceCodeVec.push_back(sourceCode);
//        #endif
//    #else
//        #ifdef HAVE_CL_CL2_HPP
//            std::vector<std::string> sourceCodeVec;
//            sourceCodeVec.push_back(sourceCode);
//        #endif
//    #endif
    
    XPtr<dynEigenMat<int> > ptrA(ptrA_);
    XPtr<dynEigenMat<int> > ptrB(ptrB_);
    XPtr<dynEigenMat<int> > ptrC(ptrC_);
    
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > refA = ptrA->data();
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > refB = ptrB->data();
    Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> > refC = ptrC->data();
    
    Eigen::Map<Eigen::MatrixXi, 0, Eigen::OuterStride<> > Am(
        refA.data(), refA.rows(), refA.cols(),
        Eigen::OuterStride<>(refA.outerStride())
    );
    Eigen::Map<Eigen::MatrixXi, 0, Eigen::OuterStride<> > Bm(
        refB.data(), refB.rows(), refB.cols(),
        Eigen::OuterStride<>(refB.outerStride())
    );
    Eigen::Map<Eigen::MatrixXi, 0, Eigen::OuterStride<> > Cm(
        refC.data(), refC.rows(), refC.cols(),
        Eigen::OuterStride<>(refC.outerStride())
    );
    
//    Rcout << "all objects initialized" << std::endl;
    
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
    
//    Rcout << "got platforms" << std::endl;
    
    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platforms[0]()),
        0
    };
    
//    Rcout << "selected default platform" << std::endl;

    // need to conditionally do CL_DEVICE_TYPE_CPU
    if(device_type == 0){
        ocl_device = CL_DEVICE_TYPE_GPU;
    }else{
        ocl_device = CL_DEVICE_TYPE_CPU;
    }
    
    Context context = createContext(ocl_device, cps, err);
        
//    Rcout << "Context Made" << std::endl;

    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    if(devices.size() < 1){
        stop("No GPU devices found");
    }
        
//    Rcout << "Device Found" << std::endl;

        
    // Create a command queue and use the first device
    CommandQueue queue = CommandQueue(context, devices[0], 0, &err);
    
//    Rcout << "Command Queue created" << std::endl;

    // Read source file - passed in by R wrapper function
//    #if defined(__APPLE__) || defined(__MACOSX)
//        #ifndef HAVE_OPENCL_CL2_HPP
//            int pl;
//            std::pair <const char*, int> sourcePair;
//            sourcePair = std::make_pair(sourceCode.c_str(), pl);
//            Program::Sources source(1, sourcePair);
//        #else
//            Program::Sources source(sourceCodeVec);
//        #endif
//    #else
//        #ifndef HAVE_CL_CL2_HPP
//            int pl;
//            std::pair <const char*, int> sourcePair;
//            sourcePair = std::make_pair(sourceCode.c_str(), pl);
//            Program::Sources source(1, sourcePair);
//        #else
//            Program::Sources source(sourceCodeVec);
//        #endif
//    #endif
//    
//    Rcout << "found kernel" << std::endl;

    // Make program of the source code in the context
    Program program = Program(context, sources);
    if (err != CL_SUCCESS)
    {
       stop("Error: Failed to create compute program!\n");
    }
    
//    Rcout << "Program created" << std::endl;
        
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
        
//    Rcout << "Program Built" << std::endl;
        
    // Make kernel
    Kernel kernel(program, "iMatMult", &err);
//        Kernel kernel(program, kernel_function, &err);
    
    if (err != CL_SUCCESS)
    {
       stop("Error: Failed to create compute kernel!\n");
    }
        
        
//        cl_int wgSize = kernel.getWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
//    Rcout << "Kernel made" << std::endl;

    // Create memory buffers
    Buffer bufferA = Buffer(context, CL_MEM_READ_ONLY, szA * sizeof(int), NULL, &err);
    Buffer bufferB = Buffer(context, CL_MEM_READ_ONLY, szB * sizeof(int), NULL, &err);
    Buffer bufferC = Buffer(context, CL_MEM_WRITE_ONLY, szC * sizeof(int), NULL, &err);
    
//    Rcout << "Buffers initialized" << std::endl;

    // Copy lists A and B to the memory buffers
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, szA * sizeof(int), Am.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, szB * sizeof(int), Bm.data());
    
//    Rcout << "Buffers written" << std::endl;

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
    
//    Rcout << "kenel args set" << std::endl;
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
//    Rcout << "kernel completed" << std::endl;
//        err = queue.enqueueNDRangeKernel(kernel, NullRange, global, NullRange);
        
    // Read buffer C into a local list        
    err = queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, 
                                szC * sizeof(int), Cm.data());
                                
//    Rcout << "Completed Buffer Read" << std::endl;
}
