#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

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
    // Get available platforms
    vector<Platform> platforms;
    Platform::get(&platforms);
    
    return(wrap(platforms.size()));
}
    
