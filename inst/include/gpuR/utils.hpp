#ifndef GPUR_UTILS
#define GPUR_UTILS

// Use OpenCL with ViennaCL
#ifdef BACKEND_CUDA
#define VIENNACL_WITH_CUDA 1
#elif defined(BACKEND_OPENCL)
#define VIENNACL_WITH_OPENCL 1
#else
#define VIENNACL_WITH_OPENCL 1
#endif

// Use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1

// ViennaCL headers
#ifndef BACKEND_CUDA
#include "viennacl/ocl/backend.hpp"
#endif

#include <Rcpp.h>

inline
std::vector<std::string> 
split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}


// Function to round down a valid to nearest 'multiple' (e.g. 16)
inline
int 
roundDown(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;
    
    int remainder = numToRound % multiple;
    if (remainder == 0 || remainder == numToRound)
        return numToRound;
    
    return numToRound - remainder;
}

inline
int 
roundUp(int numToRound, int multiple)
{
	if (multiple == 0)
		return numToRound;
	
	int remainder = numToRound % multiple;
	if (remainder == 0 || multiple == numToRound)
		return numToRound;
	
	return numToRound + multiple - remainder;
}

#ifndef BACKEND_CUDA
inline
void 
check_max_size(viennacl::ocl::context ctx, std::string kernel_name, unsigned int &max_local_size)
{
    cl_device_type type_check = ctx.current_device().type();
    
    if(type_check & CL_DEVICE_TYPE_CPU){
        max_local_size = 1;
    }else{
        cl_device_id raw_device = ctx.current_device().id();
        cl_kernel raw_kernel = ctx.get_kernel("my_kernel", kernel_name).handle().get();
        size_t preferred_work_group_size_multiple;
        
        cl_int err = clGetKernelWorkGroupInfo(raw_kernel, raw_device, 
                                              CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                              sizeof(size_t), &preferred_work_group_size_multiple, NULL);
        
        if(err != CL_SUCCESS){
            Rcpp::stop("clGetKernelWorkGroupInfo failed");
        }
        
        max_local_size = roundDown(max_local_size, preferred_work_group_size_multiple);
    }
}
#endif

#endif
