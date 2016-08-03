
#include "gpuR/windows_check.hpp"
#include "gpuR/utils.hpp"
#include "gpuR/cl_helpers.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1

// ViennaCL headers
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/platform.hpp"

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;

//' @title Detect Number of Platforms
//' @description Find out how many OpenCL enabled platforms are available.
//' @return An integer value representing the number of platforms available.
//' @seealso \link{detectGPUs}
//' @export
// [[Rcpp::export]]
SEXP detectPlatforms()
{
    typedef std::vector< viennacl::ocl::platform > platforms_type;
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
    typedef std::vector< viennacl::ocl::platform > platforms_type;
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
    
    // subtract one for zero indexing
    unsigned int platform_idx = as<unsigned int>(platform_idx_) - 1;

    // Discover number of platforms
    std::vector<cl::Platform> platforms;
    getPlatforms(platforms); // cl_helpers.hpp
    
    if(platforms.size() == 0){
        stop("No platforms found! Check OpenCL installation");
    }
    
    Platform plat = platforms[platform_idx];
    
    std::string platformName = plat.getInfo<CL_PLATFORM_NAME>();
    std::string platformVendor = plat.getInfo<CL_PLATFORM_VENDOR>();
    std::string platformVersion = plat.getInfo<CL_PLATFORM_VERSION>();
    std::string platformExtensions = plat.getInfo<CL_PLATFORM_EXTENSIONS>();
    
    std::vector<std::string> extensionsVector;
    extensionsVector = split(platformExtensions, ' ');

    return List::create(Named("platformName") = platformName,
                        Named("platformVendor") = platformVendor,
                        Named("platformVersion") = platformVersion,
                        Named("platformExtensions") = extensionsVector
                        );
}
