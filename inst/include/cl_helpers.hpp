#ifndef CL_HELPERS
#define CL_HELPERS

#include <memory>

#include <boost/scoped_ptr.hpp> //scoped_ptr

#include <CL/cl.hpp>

#include <Rcpp.h>

using namespace cl;
using namespace Rcpp;

/* C++ get platforms function to return
 * a user friendly error message if fails.
 */
inline
void getPlatforms(std::vector<Platform> &platforms)
{
    try
    {
        Platform::get(&platforms);
    }
    catch (cl::Error error)
    {
        stop("No platforms detected.  Verify your SDK installation.");
    }
}

/* C++ create context function to return
 * a user friendly error message if it fails.
 */
inline
Context createContext(
    cl_device_type deviceType, 
    cl_context_properties* cps,
    cl_int err)
    {
        boost::scoped_ptr<Context> p_context;
        
        try
        {
            p_context.reset(new Context( deviceType, cps, NULL, NULL, &err));
        }
        catch (cl::Error error)
        {
            switch(error.err()){
                case CL_INVALID_PLATFORM:
                    stop("Platform not found or not valid");
                case CL_INVALID_PROPERTY:
                    stop("Unsupported property name");
                case CL_INVALID_DEVICE_TYPE:
                    stop("Invalid device type");
                case CL_DEVICE_NOT_AVAILABLE:
                    stop("Device not currently available");
                case CL_DEVICE_NOT_FOUND:
                    stop("Device not found");
                case CL_OUT_OF_RESOURCES:
                    stop("Unable to allocate sufficient resources on device");
                case CL_OUT_OF_HOST_MEMORY:
                    stop("Unable to allocate sufficient resources on the host");
                default:
                    stop("unrecognized error");
            }
            stop("program failed to build");
        }
    
    Context& context = *p_context;
    
    return context;
}

/* Function to evaluate the err returned by clCreateContext (C OpenCL API).
 * It provides a user friendly error message if the context failed and
 * automatically releases the previous context.
 */
inline
cl_context c_createContext(
    cl_context_properties* props, 
    cl_device_id device,
    cl_int err
    )
{
    cl_context ctx = 0;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);

    if (err != CL_SUCCESS) {
        switch(err){
            case CL_INVALID_PLATFORM:
                clReleaseContext(ctx);
                stop("Platform not found or not valid");
            case CL_INVALID_PROPERTY:
                clReleaseContext(ctx);
                stop("Unsupported property name");
            case CL_INVALID_DEVICE_TYPE:
                clReleaseContext(ctx);
                stop("Invalid device type");
            case CL_DEVICE_NOT_AVAILABLE:
                clReleaseContext(ctx);
                stop("Device not currently available");
            case CL_DEVICE_NOT_FOUND:
                clReleaseContext(ctx);
                stop("Device not found");
            case CL_OUT_OF_RESOURCES:
                clReleaseContext(ctx);
                stop("Unable to allocate sufficient resources on device");
            case CL_OUT_OF_HOST_MEMORY:
                clReleaseContext(ctx);
                stop("Unable to allocate sufficient resources on the host");
            default:
                clReleaseContext(ctx);
                stop("unrecognized error");
        }
    }
    
    return ctx;
}

#endif
