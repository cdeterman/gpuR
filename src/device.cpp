
#include "gpuR/windows_check.hpp"
#include <boost/algorithm/string.hpp>
#include "gpuR/cl_helpers.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;

// [[Rcpp::export]]
SEXP cpp_detectGPUs(SEXP platform_idx)
{    
    // subtract one for zero indexing
    unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
    
    // Get available platforms
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    if(platforms.size() == 0){
        stop("No platforms found. Check OpenCL installation!\n");
    } 
        
    if (plat_idx > platforms.size()){
        stop("platform index greater than number of platforms.");
    }

    // Select context used only for GPUs
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
    viennacl::ocl::switch_context(id);

    // Select the platform
    viennacl::ocl::set_context_platform_index(id, plat_idx);
    
    // get the devices
    std::vector< viennacl::ocl::device > devices;
    devices = viennacl::ocl::current_context().devices();
    
    return(wrap(devices.size()));
}

// [[Rcpp::export]]
List cpp_gpuInfo(SEXP platform_idx_, SEXP gpu_idx_)
{
    // declarations
    cl_int err = 0;
    
    // subtract one for zero indexing
    unsigned int platform_idx = as<unsigned int>(platform_idx_) - 1;
    unsigned int gpu_idx = as<unsigned int>(gpu_idx_) - 1;    
    
    // Get available platforms
    std::vector<Platform> platforms;
    getPlatforms(platforms); // cl_helpers.hpp
    
    if(platforms.size() == 0){
        stop("No platforms found. Check OpenCL installation!");
    }

    if (platform_idx > platforms.size()){
        stop("platform index greater than number of platforms.");
    }

    // Select the platform and create a context using this platform 
    // and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[platform_idx])(),
        0
    };

    Context context( CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    if(err != CL_SUCCESS){
        stop("context failed to create"); 
    }
    
    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    Device working_device=devices[gpu_idx];
    std::string deviceName = working_device.getInfo<CL_DEVICE_NAME>();
    std::string deviceVendor = working_device.getInfo<CL_DEVICE_VENDOR>();
    cl_uint numCores = working_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cl_long amountOfMemory = working_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    cl_uint clockFreq = working_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    cl_ulong localMem = working_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    cl_ulong maxAlocatableMem = working_device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    cl_bool available = working_device.getInfo<CL_DEVICE_AVAILABLE>();
    cl_uint maxWorkGroupSize = working_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    cl_uint maxWorkItemDim = working_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    std::vector<std::size_t> maxWorkItemSizes = working_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    std::string deviceExtensions = working_device.getInfo<CL_DEVICE_EXTENSIONS>();


    std::vector<std::string> extensionsVector;
    boost::split(extensionsVector, deviceExtensions, boost::is_any_of(" "));
    std::string available_str = (available == 1) ? "yes" : "no";
    
    //Named("maxWorkItemSizes") = maxWorkItemSizes,
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
                        Named("deviceExtensions") = extensionsVector);
}


// [[Rcpp::export]]
void setGPU(SEXP gpu_idx_)
{
    // subtract one for zero indexing
    unsigned int gpu_idx = as<unsigned int>(gpu_idx_) - 1;
    
    //use only GPUs
    long id = 0;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::gpu_tag());
//    viennacl::ocl::switch_context(id);
    
    viennacl::ocl::get_context(id).switch_device(gpu_idx);
}

//// [[Rcpp::export]]
//SEXP currentDevice()
//{
////    return wrap(viennacl::ocl::current_device().name());
//    
//    return wrap(viennacl::ocl::current_context().current_device_id_);
//}

//// [[Rcpp::export]]
//SEXP cpp_detectGPUs(SEXP platform_idx)
//{
//    // declarations
//    cl_int err = 0;
//    
//    // subtract one for zero indexing
//    unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
//    
//    // Get available platforms
//    std::vector<Platform> platforms;
//    getPlatforms(platforms); // cl_helpers.hpp
//    
//    if(platforms.size() == 0){
//        stop("No platforms found. Check OpenCL installation!\n");
//    } 
//        
//    if (plat_idx > platforms.size()){
//        stop("platform index greater than number of platforms.");
//    }
//
//    // Select the platform and create a context using this platform 
//    // and the GPU
//    cl_context_properties cps[3] = {
//        CL_CONTEXT_PLATFORM,
//        (cl_context_properties)(platforms[plat_idx])(),
//        0
//    };
//    
//    Context context = createContext(CL_DEVICE_TYPE_GPU, cps, err);
//
//    // Get a list of devices on this platform
//    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
//    
//    return(wrap(devices.size()));
//}


// [[Rcpp::export]]
SEXP cpp_detectCPUs(SEXP platform_idx)
{    
    // subtract one for zero indexing
    unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
    
    // Get available platforms
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    if(platforms.size() == 0){
        stop("No platforms found. Check OpenCL installation!\n");
    } 
        
    if (plat_idx > platforms.size()){
        stop("platform index greater than number of platforms.");
    }

    // Select context used only for CPUs
    long id = 1;
    viennacl::ocl::set_context_device_type(id, viennacl::ocl::cpu_tag());
    viennacl::ocl::switch_context(id);

    // Select the platform
    viennacl::ocl::set_context_platform_index(id, plat_idx);
    
    // get the devices
    std::vector< viennacl::ocl::device > devices;
    devices = viennacl::ocl::current_context().devices();
    
    return(wrap(devices.size()));
}

//// [[Rcpp::export]]
//SEXP cpp_detectCPUs(SEXP platform_idx)
//{
//    // declarations
//    cl_int err = 0;
//    
//    // subtract one for zero indexing
//    unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
//    
//    // Get available platforms
//    std::vector<Platform> platforms;
//    getPlatforms(platforms); // cl_helpers.hpp
//    
//    if(platforms.size() == 0){
//        stop("No platforms found. Check OpenCL installation!\n");
//    } 
//        
//    if (plat_idx > platforms.size()){
//        stop("platform index greater than number of platforms.");
//    }
//
//    // Select the platform and create a context using this platform
//    cl_context_properties cps[3] = {
//        CL_CONTEXT_PLATFORM,
//        (cl_context_properties)(platforms[plat_idx])(),
//        0
//    };
//    
//    Context context = createContext(CL_DEVICE_TYPE_CPU, cps, err);
//
//    // Get a list of devices on this platform
//    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
//    
//    return(wrap(devices.size()));
//}
    
//[[Rcpp::export]]
bool cpp_device_has_double(SEXP platform_idx_, SEXP gpu_idx_){
    // declarations
    cl_int err = 0;
//    std::vector<std::string> extensionsVector;
    std::string deviceExtensions;
    
    // also check for cl_amd_fp64
    std::string double_str = "cl_khr_fp64";
    bool double_check;
    
    // subtract one for zero indexing
    unsigned int platform_idx = as<unsigned int>(platform_idx_) - 1;
    unsigned int gpu_idx = as<unsigned int>(gpu_idx_) - 1;    
    
    // Get available platforms
    std::vector<Platform> platforms;
    getPlatforms(platforms); // cl_helpers.hpp
    
    if(platforms.size() == 0){
        stop("No platforms found. Check OpenCL installation!");
    }

    if (platform_idx > platforms.size()){
        stop("platform index greater than number of platforms.");
    }

    // Select the platform and create a context using this platform 
    // and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[platform_idx])(),
        0
    };

    Context context( CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    if(err != CL_SUCCESS){
        stop("context failed to create"); 
    }
    
    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    Device working_device=devices[gpu_idx];
    
    deviceExtensions = working_device.getInfo<CL_DEVICE_EXTENSIONS>();

//    boost::split(extensionsVector, deviceExtensions, boost::is_any_of(" "));
    
    double_check = boost::contains(deviceExtensions, double_str);
    return double_check;
}

    
