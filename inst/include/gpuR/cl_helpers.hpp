#ifndef CL_HELPERS
#define CL_HELPERS

// include OpenCL C Headers
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// define options for OpenCL C++ API
#if defined(__APPLE__) || defined(__MACOSX)
    #ifdef HAVE_OPENCL_CL2_HPP
        #define CL_HPP_ENABLE_EXCEPTIONS
        #include <OpenCL/cl2.hpp>
    #else
        #define __CL_ENABLE_EXCEPTIONS
        //#define CL_USE_DEPRECATED_OPENCL_1_1_APIS    
        #include <OpenCL/cl.hpp>
    #endif

#else
    #ifdef HAVE_CL_CL2_HPP
        #define CL_HPP_ENABLE_EXCEPTIONS
        #include <CL/cl2.hpp>
    #else
        #define __CL_ENABLE_EXCEPTIONS
        // #define CL_USE_DEPRECATED_OPENCL_1_1_APIS        
        #include <CL/cl.hpp>
    #endif
#endif


#include <memory>
#include <Rcpp.h>

//using namespace cl;
using namespace Rcpp;

/* C++ get platforms function to return
 * a user friendly error message if fails.
 */
inline
void getPlatforms(std::vector<cl::Platform> &platforms)
{
    try
    {
        cl::Platform::get(&platforms);
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
cl::Context createContext(
    cl_device_type deviceType, 
    cl_context_properties* cps,
    cl_int err)
    {
        std::unique_ptr<cl::Context> p_context;

        try
        {
            p_context.reset(new cl::Context( deviceType, cps, NULL, NULL, &err));
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
    
    cl::Context& context = *p_context;
    
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

inline
unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#endif
