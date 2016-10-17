
#include "gpuR/windows_check.hpp"
#include "gpuR/utils.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// ViennaCL headers
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/platform.hpp"

#include <Rcpp.h>

using namespace Rcpp;

typedef std::vector< viennacl::ocl::platform > platforms_type;

//' @title Detect Number of Platforms
//' @description Find out how many OpenCL enabled platforms are available.
//' @return An integer value representing the number of platforms available.
//' @seealso \link{detectGPUs}
//' @export
// [[Rcpp::export]]
SEXP detectPlatforms()
{
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    return wrap(platforms.size());
}

//// [[Rcpp::export]]
//void setPlatform(SEXP platform_idx_)
//{
//    unsigned int platform_idx = as<unsigned int>(platform_idx_);
//    typedef std::vector< viennacl::ocl::platform > platforms_type;
//    
//    // get platforms
//    platforms_type platforms = viennacl::ocl::get_platforms();
//    unsigned int platforms_size = platforms.size();
//    
//    if(platform_idx > platforms_size){
//        stop("Platform index out of bounds");
//    }
//    
//    // subtract one for zero indexing    
//    // set platform
//    long id = 0;
//    viennacl::ocl::set_context_platform_index(id, platform_idx - 1);
//}

//' @title Return Current Platform
//' @description Find out which platform is currently in use
//' @return \item{platform}{Name of the current platform}
//' @return \item{platform_index}{Index of current platform}
//' @seealso \link{detectPlatforms}
//' @export
// [[Rcpp::export]]
SEXP currentPlatform()
{
    // get current platform index
    int plat_idx = viennacl::ocl::current_context().platform_index();
    
    // get platforms
    platforms_type platforms = viennacl::ocl::get_platforms();
    
//    return wrap(plat_idx + 1);
//    return wrap(platforms[plat_idx].info());
    
    return List::create(Named("platform") = wrap(platforms[plat_idx].info()),
                        Named("platform_index") = wrap(plat_idx + 1));
}

//List platformNames()
//{
//    
//    std::cout << platforms[0].info();
//}

//std::vector<std::string> split(const std::string &s, char delim) {
//    std::vector<std::string> elems;
//    split(s, delim, elems);
//    return elems;
//}


// [[Rcpp::export]]
List cpp_platformInfo(SEXP platform_idx_)
{
    cl_int err;
    
    // get platforms
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    // subtract one for zero indexing
    unsigned int platform_idx = as<unsigned int>(platform_idx_) - 1;
    
    viennacl::ocl::platform vcl_platform = platforms[platform_idx];
    
    cl_platform_id platform_id = vcl_platform.id();
    
    char platformName[1024];
    char platformVendor[1024];
    char platformVersion[1024];
    char platformExtensions[2048];
        
    err = clGetPlatformInfo(platform_id, 
                            CL_PLATFORM_NAME,
                            sizeof(platformName),
                            platformName,
                            NULL
    );
    err = clGetPlatformInfo(platform_id, 
                            CL_PLATFORM_VENDOR,
                            sizeof(platformName),
                            platformVendor,
                            NULL
    );
    err = clGetPlatformInfo(platform_id, 
                            CL_PLATFORM_VERSION,
                            sizeof(platformName),
                            platformVersion,
                            NULL
    );
    err = clGetPlatformInfo(platform_id, 
                            CL_PLATFORM_EXTENSIONS,
                            sizeof(platformName),
                            platformExtensions,
                            NULL
    );
    
    // Convert char arrays to string
    std::string platformNameStr(platformName);
    std::string platformVendorStr(platformVendor);
    std::string platformVersionStr(platformVersion);

    // Split extensions to a vector
    std::vector<std::string> extensionsVector;
    std::string platformExtensionsStr(platformExtensions);
    extensionsVector = split(platformExtensionsStr, ' ');

    return List::create(Named("platformName") = platformNameStr,
                        Named("platformVendor") = platformVendorStr,
                        Named("platformVersion") = platformVersionStr,
                        Named("platformExtensions") = extensionsVector
                        );
}
