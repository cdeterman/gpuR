#ifndef OPENCL_UTILS
#define OPENCL_UTILS

inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
        << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif