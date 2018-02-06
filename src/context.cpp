#include "gpuR/windows_check.hpp"
// #include "gpuR/cl_helpers.hpp"

// Use OpenCL with ViennaCL
#ifdef BACKEND_CUDA
#define VIENNACL_WITH_CUDA 1
#elif defined(BACKEND_OPENCL)
#define VIENNACL_WITH_OPENCL 1
#else
#define VIENNACL_WITH_OPENCL 1
#endif
//#define VIENNACL_DEBUG_ALL 1

// ViennaCL headers
#ifndef BACKEND_CUDA
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"
#endif

#include <Rcpp.h>

// using namespace cl;
using namespace Rcpp;

#ifdef BACKEND_CUDA


int currentContext()
{
    int cuda_device;
    cudaGetDevice(&cuda_device);
    return cuda_device + 1;
}


void
cpp_setContext(int id)
{
    if(id <= 0){
        stop("Index cannot be 0 or less");
    }
    cudaSetDevice(id - 1);
}


void initContexts(){
    // declarations
    int id = 0;
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    for(unsigned int gpu_idx = 0; gpu_idx < device_count; gpu_idx++) {
        cudaDeviceProp prop;    
        Rcpp::Rcout << "  - context device index: " << gpu_idx << std::endl;
        cudaGetDeviceProperties(&prop, gpu_idx);
        Rcpp::Rcout << "    - " << prop.name << std::endl;
        
        // increment context
        id++;
    }
    
    Rcpp::Rcout << "checked all devices" << std::endl;
    Rcpp::Rcout << "completed initialization" << std::endl;
}


DataFrame 
listContexts()
{
    
    // declarations
    int id = 0;
    int num_contexts = 0;
    cudaGetDeviceCount(&num_contexts);
    
    Rcpp::IntegerVector   context_index(num_contexts);
    Rcpp::CharacterVector platform_name(num_contexts);
    Rcpp::IntegerVector   platform_index(num_contexts);
    Rcpp::CharacterVector device_name(num_contexts);
    Rcpp::IntegerVector   device_index(num_contexts);
    Rcpp::CharacterVector device_type(num_contexts);
    
    for(unsigned int gpu_idx = 0; gpu_idx < num_contexts; gpu_idx++) {
        
        //            Rcout << "context id" << std::endl;
        //            Rcout << id << std::endl;
        
        //            Rcout << "current platform index" << std::endl;
        //            Rcout << plat_idx << std::endl;
        
        context_index[id] = id + 1;
        platform_index[id] = 0;
        platform_name[id] = "NVIDIA platform";
        
        //            viennacl::ocl::set_context_platform_index(id, plat_idx);
        //            
        ////            Rcout << "set platform successfully" << std::endl;
        //            
        //            // Select device
        //            viennacl::ocl::setup_context(id, devices[gpu_idx]);
        //            
        ////            Rcout << "switched device successfully" << std::endl;
        
        // Get device info
        cudaDeviceProp prop;    
        cudaGetDeviceProperties(&prop, gpu_idx);
        device_index[id] = 0;
        device_name[id] = prop.name;
        device_type[id] = "gpu";
        
        // increment context
        id++;
    }
    
    return Rcpp::DataFrame::create(Rcpp::Named("context") = context_index,
                                   Rcpp::Named("platform") = platform_name,
                                   Rcpp::Named("platform_index") = platform_index,
                                   Rcpp::Named("device") = device_name,
                                   Rcpp::Named("device_index") = device_index,
                                   Rcpp::Named("device_type") = device_type,
                                   _["stringsAsFactors"] = false );
}


SEXP
getContextPtr(const int ctx_id){
    Rcpp::stop("Contexts are not relevant for CUDA backend");
}

#else

// [[Rcpp::export]]
void initContexts(){
    // declarations
    int id = 0;
    
    // get platforms
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    Rcpp::Rcout << "Number of platforms: " << platforms.size() << std::endl;
    
    for(unsigned int plat_idx = 0; plat_idx < platforms.size(); plat_idx++) {
        
        Rcpp::Rcout << "- platform: " << platforms[plat_idx].info() << std::endl;
    
        std::vector< viennacl::ocl::device > devices;
        devices = platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL);
    
        for(unsigned int gpu_idx = 0; gpu_idx < devices.size(); gpu_idx++) {
            
            Rcpp::Rcout << "  - context device index: " << gpu_idx << std::endl;
            viennacl::ocl::set_context_platform_index(id, plat_idx);
            viennacl::ocl::setup_context(id, devices[gpu_idx]);
            Rcpp::Rcout << "    - " << devices[gpu_idx].name() << std::endl;
            
            // increment context
            id++;
        }
    }
    
    Rcpp::Rcout << "checked all devices" << std::endl;
    
    viennacl::ocl::switch_context(0);
    
    Rcpp::Rcout << "completed initialization" << std::endl;
}


//' @title Available OpenCL Contexts
//' @description Provide a data.frame of available OpenCL contexts and
//' associated information.
//' @return data.frame containing the following fields
//' @return \item{context}{Integer identifying context}
//' @return \item{platform}{Character string listing OpenCL platform}
//' @return \item{platform_index}{Integer identifying platform}
//' @return \item{device}{Character string listing device name}
//' @return \item{device_index}{Integer identifying device}
//' @return \item{device_type}{Character string labeling device (e.g. gpu)}
//' @export
// [[Rcpp::export]]
DataFrame 
listContexts()
{

    // declarations
    int id = 0;
    int num_contexts = 0;
    //int num_devices;
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    
    // get platforms
    platforms_type platforms = viennacl::ocl::get_platforms();  
    
//    Rcout << "number of platforms found" << std::endl;
//    Rcout << platforms.size() << std::endl;
    
    // count number of contexts initialized
    // for each platform    
    for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
        
        num_contexts += platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL).size();
//        for(unsigned int gpu_idx=0; gpu_idx < platforms[plat_idx].devices().size(); gpu_idx++){
//            num_contexts++;
//        }
    }
    
    // need to multiply by number of platforms too
    // num_contexts *= platforms.size();
    
//    Rcout << "number of total contexts to create" << std::endl;
//    Rcout << num_contexts << std::endl;
    
    Rcpp::IntegerVector   context_index(num_contexts);
    Rcpp::CharacterVector platform_name(num_contexts);
    Rcpp::IntegerVector   platform_index(num_contexts);
    Rcpp::CharacterVector device_name(num_contexts);
    Rcpp::IntegerVector   device_index(num_contexts);
    Rcpp::CharacterVector device_type(num_contexts);
    
    for(unsigned int plat_idx = 0; plat_idx < platforms.size(); plat_idx++) {
        
        std::vector< viennacl::ocl::device > devices;
        devices = platforms[plat_idx].devices(CL_DEVICE_TYPE_ALL);
        
        for(unsigned int gpu_idx = 0; gpu_idx < devices.size(); gpu_idx++) {
        
//            Rcout << "context id" << std::endl;
//            Rcout << id << std::endl;
            
//            Rcout << "current platform index" << std::endl;
//            Rcout << plat_idx << std::endl;
            
            context_index[id] = id + 1;
            platform_index[id] = plat_idx;
            platform_name[id] = platforms[plat_idx].info();
            
//            viennacl::ocl::set_context_platform_index(id, plat_idx);
//            
////            Rcout << "set platform successfully" << std::endl;
//            
//            // Select device
//            viennacl::ocl::setup_context(id, devices[gpu_idx]);
//            
////            Rcout << "switched device successfully" << std::endl;
            
            // Get device info
            device_index[id] = 0;
            device_name[id] = devices[gpu_idx].name();
            // device_name[id] = viennacl::ocl::current_device().name();
            
//            Rcout << "current device index" << std::endl;
//            Rcout << device_index[id] << std::endl;
            
//            Rcout << "current device name" << std::endl;
//            Rcout << device_name[id] << std::endl;
            

	    cl_device_type check = devices[gpu_idx].type(); 
	    if(check & CL_DEVICE_TYPE_CPU){
		device_type[id] = "cpu";
	    }else if(check & CL_DEVICE_TYPE_GPU){
		device_type[id] = "gpu";
	    }else if(check & CL_DEVICE_TYPE_ACCELERATOR){
		device_type[id] = "accelerator";
	    }else{
		Rcpp::Rcout << "device found: " << std::endl;
		Rcpp::Rcout << check << std::endl;
		throw Rcpp::exception("unrecognized device detected");
 
	    }

	    // increment context
            id++;
        }
    }
    
    return Rcpp::DataFrame::create(Rcpp::Named("context") = context_index,
    			  Rcpp::Named("platform") = platform_name,
                  Rcpp::Named("platform_index") = platform_index,
				  Rcpp::Named("device") = device_name,
                  Rcpp::Named("device_index") = device_index,
                  Rcpp::Named("device_type") = device_type,
                  _["stringsAsFactors"] = false );
}


//' @title Current Context
//' @description Get current context index
//' @return An integer reflecting the context listed in \link{listContexts}
//' @export
// [[Rcpp::export]]
int currentContext()
{
    return viennacl::ocl::backend<>::current_context_id() + 1;
}


// [[Rcpp::export]]
void
cpp_setContext(int id)
{
    if(id <= 0){
        stop("Index cannot be 0 or less");
    }
    viennacl::ocl::switch_context(id - 1);
}

// [[Rcpp::export]]
SEXP
getContextPtr(const int ctx_id){
    
    viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
    Rcpp::XPtr<viennacl::ocl::context> ptrctx(&ctx);
    return ptrctx;
}

#endif
