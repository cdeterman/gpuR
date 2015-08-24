
#include <boost/algorithm/string.hpp>

#include "gpuR/cl_helpers.hpp"

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;


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
