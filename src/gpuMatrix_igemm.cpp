
#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/dynEigenMat.hpp"
#include "gpuR/cl_helpers.hpp"

using namespace cl;
using namespace Rcpp;

// static const char * my_compute_program =
//  "__kernel void elementwise_prod(\n"
//  "          __global const int * vec1,\n"
//  "          __global const int * vec2, \n"
//  "          __global int * result,\n"
//  "          unsigned int size) \n"
//  "{ \n"
//  "  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
//  "      for (unsigned int j = get_global_id(1); j < size; j += get_global_size(1))\n"
//  "          result[i] = vec1[i] * vec2[i];\n"
//  "};\n";

static const char * my_compute_program =
    "__kernel void elementwise_prod(\n"
    "          __global const int * vec1,\n"
    "          __global const int * vec2, \n"
    "          __global int * result,\n"
    "          unsigned int size) \n"
    "{ \n"
    "  unsigned int i = get_global_id(0);\n"
    "  result[i] = vec1[i] * vec2[i];\n"
    "};\n";

//[[Rcpp::export]]
void 
cpp_gpuMatrix_custom_igemm(
        SEXP ptrA_, SEXP ptrB_, SEXP ptrC_,
        SEXP sourceCode_,
        int ctx_id)
{
    std::string my_kernel = as<std::string>(sourceCode_);
    
    XPtr<dynEigenMat<int> > ptrA(ptrA_);
    XPtr<dynEigenMat<int> > ptrB(ptrB_);
    XPtr<dynEigenMat<int> > ptrC(ptrC_);
    
    // move data to device
    viennacl::matrix<int> vcl_A = ptrA->device_data(ctx_id);
    viennacl::matrix<int> vcl_B = ptrB->device_data(ctx_id);
    viennacl::matrix<int> vcl_C = ptrC->device_data(ctx_id);
    
    int M = vcl_A.size2();
    // int N = vcl_B.size1();
    int P = vcl_A.size1();
    int M_internal = vcl_A.internal_size2();
    int P_internal = vcl_A.internal_size1();
    
    // std::cout << "internal rows" << std::endl;
    // std::cout << vcl_A.internal_size1() << std::endl;
    // std::cout << "internal cols" << std::endl;
    // std::cout << vcl_A.internal_size2() << std::endl;
    
    // add kernel to program
    viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(my_kernel, "my_kernel");
    // viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(my_compute_program, "my_compute_program");
    
    // get compiled kernel function
    viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("iMatMult");
    // viennacl::ocl::kernel & my_kernel_mul = my_prog.get_kernel("elementwise_prod");

    // execute kernel
    viennacl::ocl::enqueue(my_kernel_mul(M, M_internal, P, P_internal, vcl_A, vcl_B, vcl_C));
    // viennacl::ocl::enqueue(my_kernel_mul(vcl_A, vcl_B, vcl_C, static_cast<cl_uint>(M + P)));
    
    std::cout << "A: " << std::endl;
    std::cout << vcl_A << std::endl;
    
    std::cout << "B: " << std::endl;
    std::cout << vcl_B << std::endl;
    
    std::cout << "C matrix" << std::endl;
    std::cout << vcl_C << std::endl;
    
    // move back to host
    ptrC->to_host(vcl_C);
}

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
    // err = kernel.setArg(1, sizeof(int), &Ndim);
    err = kernel.setArg(1, sizeof(int), &Pdim);

    err = kernel.setArg(2, bufferA);
    err = kernel.setArg(3, bufferB);
    err = kernel.setArg(4, bufferC);

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
