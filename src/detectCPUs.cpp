
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef CL_VERSION_1_3
#undef CL_VERSION_1_3
#endif

#ifdef CL_VERSION_1_2
#undef CL_VERSION_1_2
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <memory>

#include "gpuR/cl_helpers.hpp"

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;

// [[Rcpp::export]]
SEXP cpp_detectCPUs(SEXP platform_idx)
{
    // declarations
    cl_int err;
    
    // subtract one for zero indexing
    unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
    
    // Get available platforms
    std::vector<Platform> platforms;
    getPlatforms(platforms); // cl_helpers.hpp
    
    if(platforms.size() == 0){
        stop("No platforms found. Check OpenCL installation!\n");
    } 
        
    if (plat_idx > platforms.size()){
        stop("platform index greater than number of platforms.");
    }

    // Select the platform and create a context using this platform
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[plat_idx])(),
        0
    };
    
    Context context = createContext(CL_DEVICE_TYPE_CPU, cps, err);

    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    return(wrap(devices.size()));
}
    
