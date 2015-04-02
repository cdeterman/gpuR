#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;

//' @export
// [[Rcpp::export]]
SEXP detectGPUs()
{
    // Get available platforms
    vector<Platform> platforms;
    Platform::get(&platforms);
    
    return(wrap(platforms.size()));
}
    
