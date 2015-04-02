
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <Rcpp.h>

#include "opencl_utils.h"

using namespace cl;
using namespace Rcpp;

// [[Rcpp::export]]
SEXP cpp_detectGPUs(SEXP platform_idx)
{
    // declarations
    cl_int err;
    
    // subtract one for zero indexing
    unsigned int plat_idx = as<unsigned int>(platform_idx) - 1;
    
    // Get available platforms
    std::vector<Platform> platforms;
    Platform::get(&platforms);
    
    checkErr(platforms.size()!=0 ? CL_SUCCESS : -1, 
        "No platforms found. Check OpenCL installation!\n");

    if (plat_idx > platforms.size()){
        std::cerr << "ERROR: platform index greater than number of platforms." 
        << std::endl;
        exit(EXIT_FAILURE);
    }

    // Select the platform and create a context using this platform 
    // and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[plat_idx])(),
        0
    };

    Context context( CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    checkErr(err, "Conext::Context()"); 

    // Get a list of devices on this platform
    std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    return(wrap(devices.size()));
}
    

