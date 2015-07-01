#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <CL/cl.hpp>

#include <boost/algorithm/string.hpp>
#include "gpuR/cl_helpers.hpp"

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;

//[[Rcpp::export]]
bool cpp_device_has_double(SEXP platform_idx_, SEXP gpu_idx_){
    // declarations
    cl_int err;
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