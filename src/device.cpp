
#include "gpuR/windows_check.hpp"
#include "gpuR/utils.hpp"
//#include "gpuR/cl_helpers.hpp"

// Use OpenCL with ViennaCL
#ifdef BACKEND_CUDA
#define VIENNACL_WITH_CUDA 1
#elif defined(BACKEND_OPENCL)
#define VIENNACL_WITH_OPENCL 1
#else
#define VIENNACL_WITH_OPENCL 1
#endif
//#define VIENNACL_DEBUG_ALL 1

// ViennaCL headers
#ifndef BACKEND_CUDA
#include "viennacl/ocl/backend.hpp"
#endif
#include "viennacl/detail/matrix_def.hpp"

#include <Rcpp.h>

//using namespace cl;
using namespace Rcpp;


// [[Rcpp::export]]
SEXP cpp_deviceType(SEXP gpu_idx_, int ctx_idx)
{
    std::string device_type;
    
#ifndef BACKEND_CUDA
    // set context
    viennacl::context ctx(viennacl::ocl::get_context(ctx_idx));
    
    unsigned int gpu_idx = (Rf_isNull(gpu_idx_)) ? ctx.opencl_context().current_device_id() : as<unsigned int>(gpu_idx_) - 1;
    
    // Get device
    cl_device_type check = ctx.opencl_context().devices()[gpu_idx].type();
    
    if(check & CL_DEVICE_TYPE_CPU){
        device_type = "cpu";
    }else if(check & CL_DEVICE_TYPE_GPU){
        device_type = "gpu";
    }else if(check & CL_DEVICE_TYPE_ACCELERATOR){
        device_type = "accelerator";
    }else{
        Rcpp::Rcout << "device found: " << std::endl;
        Rcpp::Rcout << check << std::endl;
        throw Rcpp::exception("unrecognized device detected");
    }
#else 
    device_type = "gpu";
#endif
    
    return(wrap(device_type));
}


// [[Rcpp::export]]
SEXP cpp_detectGPUs(SEXP platform_idx)
{    
    int device_count = 0;
    
#ifndef BACKEND_CUDA
    // Get available platforms
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    typedef std::vector< viennacl::ocl::device > device_type;
    device_type devices;
    
    if(Rf_isNull(platform_idx)){
        for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
            
            devices = platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL);
            for(unsigned int device_idx=0; device_idx < devices.size(); device_idx++){
                if(devices[device_idx].type() & CL_DEVICE_TYPE_GPU){
                    device_count++;
                }
            }
        }
    }else{
        // subtract one for zero indexing
        unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
        
        devices = platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL);
        for(unsigned int device_idx=0; device_idx < devices.size(); device_idx++){
            if(devices[device_idx].type() & CL_DEVICE_TYPE_GPU){
                device_count++;
            }
        }
    }    
#else
    cudaGetDeviceCount(&device_count);
#endif
    
    return(wrap(device_count));
}


// [[Rcpp::export]]
List cpp_gpuInfo(SEXP gpu_idx_, int ctx_idx)
{

#ifndef BACKEND_CUDA
    // set context
    viennacl::context ctx(viennacl::ocl::get_context(ctx_idx));
    
    unsigned int gpu_idx = (Rf_isNull(gpu_idx_)) ? ctx.opencl_context().current_device_id() : as<unsigned int>(gpu_idx_) - 1;
    
    // Get device
    viennacl::ocl::device working_device = ctx.opencl_context().devices()[gpu_idx];
    
    std::string deviceName = working_device.name();
    std::string deviceVendor = working_device.vendor();
    cl_uint numCores = working_device.max_compute_units();
    cl_long amountOfMemory = working_device.global_mem_size();
    cl_uint clockFreq = working_device.max_clock_frequency();
    cl_ulong localMem = working_device.local_mem_size();
    cl_ulong maxAlocatableMem = working_device.max_mem_alloc_size();
    cl_bool available = working_device.available();
    cl_uint maxWorkGroupSize = working_device.max_work_group_size();
    cl_uint maxWorkItemDim = working_device.max_work_item_dimensions();
    std::vector<std::size_t> maxWorkItemSizes = working_device.max_work_item_sizes();
    std::string deviceExtensions = working_device.extensions();
    bool double_support = working_device.double_support();


    std::vector<std::string> extensionsVector;
    extensionsVector = split(deviceExtensions, ' ');
    std::string available_str = (available == 1) ? "yes" : "no";
    
    return List::create(Named("deviceName") = deviceName,
                        Named("deviceVendor") = deviceVendor,
                        Named("numberOfCores") = numCores,
                        Named("maxWorkGroupSize") = maxWorkGroupSize,
                        Named("maxWorkItemDim") = maxWorkItemDim,
                        Named("maxWorkItemSizes") = maxWorkItemSizes,
                        Named("deviceMemory") = amountOfMemory,
                        Named("clockFreq") = clockFreq,
                        Named("localMem") = localMem,
                        Named("maxAllocatableMem") = maxAlocatableMem,
                        Named("available") = available_str,
                        Named("deviceExtensions") = extensionsVector,
                        Named("double_support") = double_support);

#else

    int gpu_idx;
    if(Rf_isNull(gpu_idx_)){
        cudaGetDevice(&gpu_idx);
    }else{
        gpu_idx = as<int>(gpu_idx_) - 1;
    }
    
    // Get device info
    cudaDeviceProp prop;    
    cudaGetDeviceProperties(&prop, gpu_idx);
    
    std::string deviceName = prop.name;
    std::string deviceVendor = "NVIDIA";
    int major = prop.major;
    int minor = prop.minor;
    int numMultiProcs = prop.multiProcessorCount;
    int sharedMemBlock = prop.sharedMemPerBlock;
    int regsPerBlock = prop.regsPerBlock;
    int warpSize = prop.warpSize;
    int totalConstMem = prop.totalConstMem;
    long amountOfMemory = prop.totalGlobalMem;
    int clockFreq = prop.clockRate;
    
    // not directly in device properties
    // according to cuda faqs (https://developer.nvidia.com/cuda-faq)
    // double precision is supported with comput >= 1.3
    bool double_support;
    
    if(major > 1){
        double_support = true;
    }else{
        if(major < 1){
            double_support = false;
        }else{
            double_support = (minor >= 3) ? true : false;
        }
    }
    
    return List::create(Named("deviceName") = deviceName,
                        Named("deviceVendor") = deviceVendor,
                        Named("majorVersion") = major,
                        Named("minorVersion") = minor,
                        Named("numberOfMultiProcs") = numMultiProcs,
                        Named("sharedMemPerBlock") = sharedMemBlock,
                        Named("regsPerBlock") = regsPerBlock,
                        Named("warpSize") = warpSize,
                        Named("deviceMemory") = amountOfMemory,
                        Named("deviceConstMemory") = totalConstMem,
                        Named("clockFreq") = clockFreq,
                        Named("double_support") = double_support);
    
#endif
        
}


// [[Rcpp::export]]
List cpp_cpuInfo(SEXP cpu_idx_, int ctx_idx)
{
    
#ifndef BACKEND_CUDA
    // set context
    viennacl::context ctx(viennacl::ocl::get_context(ctx_idx));
    
    unsigned int cpu_idx = (Rf_isNull(cpu_idx_)) ? ctx.opencl_context().current_device_id() : as<unsigned int>(cpu_idx_) - 1;
    
    // Get device
    viennacl::ocl::device working_device = ctx.opencl_context().devices()[cpu_idx];
    
    if(working_device.type() & CL_DEVICE_TYPE_CPU){
	// do nothing
    }else{
        stop("device is not a CPU");
    }
    
    std::string deviceName = working_device.name();
    std::string deviceVendor = working_device.vendor();
    cl_uint numCores = working_device.max_compute_units();
    cl_long amountOfMemory = working_device.global_mem_size();
    cl_uint clockFreq = working_device.max_clock_frequency();
    cl_ulong localMem = working_device.local_mem_size();
    cl_ulong maxAlocatableMem = working_device.max_mem_alloc_size();
    cl_bool available = working_device.available();
    cl_uint maxWorkGroupSize = working_device.max_work_group_size();
    cl_uint maxWorkItemDim = working_device.max_work_item_dimensions();
    std::vector<std::size_t> maxWorkItemSizes = working_device.max_work_item_sizes();
    std::string deviceExtensions = working_device.extensions();
    bool double_support = working_device.double_support();

    std::vector<std::string> extensionsVector;
    extensionsVector = split(deviceExtensions, ' ');
    std::string available_str = (available == 1) ? "yes" : "no";
    
    return List::create(Named("deviceName") = deviceName,
                        Named("deviceVendor") = deviceVendor,
                        Named("numberOfCores") = numCores,
                        Named("maxWorkGroupSize") = maxWorkGroupSize,
                        Named("maxWorkItemDim") = maxWorkItemDim,
                        Named("maxWorkItemSizes") = maxWorkItemSizes,
                        Named("deviceMemory") = amountOfMemory,
                        Named("clockFreq") = clockFreq,
                        Named("localMem") = localMem,
                        Named("maxAllocatableMem") = maxAlocatableMem,
                        Named("available") = available_str,
                        Named("deviceExtensions") = extensionsVector,
                        Named("double_support") = double_support);
#else
    Rcpp::stop("NVIDIA backend compiled, no CPU supported");
#endif
}



//// [[Rcpp::export]]
//void setGPU(SEXP gpu_idx_)
//{
//    // subtract one for zero indexing
//    unsigned int gpu_idx = as<unsigned int>(gpu_idx_) - 1;
//    
//    //use only GPUs
//    long id = 0;
//    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
////    viennacl::ocl::switch_context(id);
//    
//    viennacl::ocl::get_context(id).switch_device(gpu_idx);
//}


//' @title Current Device Information
//' @description Check current device information
//' @return list containing
//' @return \item{device}{Character string of device name}
//' @return \item{device_index}{Integer identifying device}
//' @return \item{device_type}{Character string identifying device type (e.g. gpu)}
//' @export
// [[Rcpp::export]]
SEXP currentDevice()
{
    std::string device_type;
    std::string device_name;
    int device_idx;
    
#ifndef BACKEND_CUDA
    
    cl_device_type check = viennacl::ocl::current_device().type(); 
    
    if(check & CL_DEVICE_TYPE_CPU){
        device_type = "cpu";
    }else if(check & CL_DEVICE_TYPE_GPU){
        device_type = "gpu";
    }else if(check & CL_DEVICE_TYPE_ACCELERATOR){
        device_type = "accelerator";
    }else{
        Rcpp::Rcout << "device found: " << std::endl;
        Rcpp::Rcout << check << std::endl;
        throw Rcpp::exception("unrecognized device detected");
        
    }
    
    device_idx = (int)(viennacl::ocl::current_context().current_device_id()) + (int)(1);
    device_name = viennacl::ocl::current_context().current_device().name();
    
#else
    // get current device index
    cudaGetDevice(&device_idx);
    
    // assign device as gpu (only thing for NVIDIA)
    device_type = "gpu";
    
    // Get device info
    cudaDeviceProp prop;    
    cudaGetDeviceProperties(&prop, device_idx);
    
    // get device name
    device_name = prop.name;
    
    // increment for reporting index base-1
    device_idx++;
    
#endif
    
    return List::create(Named("device") = wrap(device_name),
                        Named("device_index") = wrap(device_idx),
                        Named("device_type") = wrap(device_type));
}



// [[Rcpp::export]]
SEXP cpp_detectCPUs(SEXP platform_idx)
{    
    
#ifndef BACKEND_CUDA
    
    int device_count = 0;
    
    // Get available platforms
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    typedef std::vector< viennacl::ocl::device > device_type;
    device_type devices;
    
    if(Rf_isNull(platform_idx)){
        for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
            
            devices = platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL);
            for(unsigned int device_idx=0; device_idx < devices.size(); device_idx++){
                if(devices[device_idx].type() & CL_DEVICE_TYPE_CPU){
                    device_count++;
                }
            }
        }
    }else{
        // subtract one for zero indexing
        unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
        
        devices = platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL);
        for(unsigned int device_idx=0; device_idx < devices.size(); device_idx++){
            if(devices[device_idx].type() & CL_DEVICE_TYPE_CPU){
                device_count++;
            }
        }
    }    
    
    return(wrap(device_count));
    
#else
    Rcpp::stop("CPUs not supported with NVIDIA backend");
#endif
}


#ifndef BACKEND_CUDA

#include "gpuR/windows_check.hpp"

#include <RcppEigen.h>

#include "viennacl/ocl/backend.hpp"

#include "gpuR/utils.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
int
preferred_wg_size(
    SEXP sourceCode_,
    std::string kernel_name,
    const int ctx_id)
{
    unsigned int max_local_size;
    
    // get kernel
    std::string my_kernel = as<std::string>(sourceCode_);
    
    // get context
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    
    // add kernel to program
    viennacl::ocl::program & my_prog = ctx.add_program(my_kernel, "my_kernel");
    
    // device type check
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = my_prog.get_kernel(kernel_name).handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("clGetKernelWorkGroupInfo failed");
        }
        
        max_local_size = preferred_work_group_size_multiple;
    }
    
    return max_local_size;
}

#endif


