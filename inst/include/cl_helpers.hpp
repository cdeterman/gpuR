#ifndef CL_HELPERS
#define CL_HELPERS

// global variable for gpu/opencl related code

// global double check
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
static int GPU_HAS_DOUBLE = 1;
#elif defined(cl_amd_fp64)
static int GPU_HAS_DOUBLE = 1;
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
static int GPU_HAS_DOUBLE = 0;
#endif

#endif
