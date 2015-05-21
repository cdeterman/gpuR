#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <boost/algorithm/string.hpp>
#include "cl_helpers.hpp"

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;


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
    boost::split(extensionsVector, platformExtensions, boost::is_any_of(" "));

    return List::create(Named("platformName") = platformName,
                        Named("platformVendor") = platformVendor,
                        Named("platformVersion") = platformVersion,
                        Named("platformExtensions") = extensionsVector
                        );
}
