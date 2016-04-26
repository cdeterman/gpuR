
#include "gpuR/windows_check.hpp"
#include <boost/algorithm/string.hpp>
//#include "gpuR/cl_helpers.hpp"

// Use OpenCL with ViennaCL
#define VIENNACL_WITH_OPENCL 1
//#define VIENNACL_DEBUG_ALL 1

// ViennaCL headers
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

#include <Rcpp.h>

//using namespace cl;
using namespace Rcpp;


// [[Rcpp::export]]
SEXP initContexts(){
    
    // declarations
    int id = 0;
    
    // get platforms
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
    
        for(unsigned int gpu_idx=0; gpu_idx < platforms[plat_idx].devices().size(); gpu_idx++){
                    
            // Select the platform
            viennacl::ocl::switch_context(id);
            viennacl::ocl::set_context_platform_index(id, plat_idx);
        
            // Get available devices
//            std::vector<viennacl::ocl::device> const & devices = viennacl::ocl::platform().devices();
            
//            std::cout << devices[gpu_idx].name() << std::endl;
            
            // take the n-th available device from 'devices'
//            std::vector< viennacl::ocl::device > my_devices;
//            my_devices.push_back(devices[gpu_idx]);
            
            // Select device
//            viennacl::ocl::setup_context(id, my_devices);
//            viennacl::ocl::current_context().switch_device(gpu_idx);
            viennacl::ocl::get_context(id).switch_device(gpu_idx);
//            std::cout << viennacl::ocl::current_context().current_device().name() << std::endl;
            
            // increment context
            id++;
        }
    }
    
    viennacl::ocl::switch_context(0);
    
//    std::cout << viennacl::ocl::current_context().current_device().name() << std::endl;
    return wrap(viennacl::ocl::current_context().current_device().name());
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
    long current_context_id = viennacl::ocl::backend<>::current_context_id();
    int num_contexts = 0;
    //int num_devices;
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    
    // get platforms
    platforms_type platforms = viennacl::ocl::get_platforms();  
    
    Rcout << "number of platforms found" << std::endl;
    Rcout << platforms.size() << std::endl;
    
    // count number of contexts initialized
    // for each platform    
    for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
        
        num_contexts += platforms[plat_idx].devices().size();
//        for(unsigned int gpu_idx=0; gpu_idx < platforms[plat_idx].devices().size(); gpu_idx++){
//            num_contexts++;
//        }
    }
    
    Rcout << "number of total contexts to create" << std::endl;
    Rcout << num_contexts << std::endl;
    
    Rcpp::IntegerVector context_index(num_contexts);
    Rcpp::CharacterVector platform_name(num_contexts);
    Rcpp::IntegerVector platform_index(num_contexts);
    Rcpp::CharacterVector device_name(num_contexts);
    Rcpp::IntegerVector device_index(num_contexts);
    Rcpp::CharacterVector device_type(num_contexts);
    
    
    for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
        
        for(unsigned int gpu_idx=0; gpu_idx < platforms[plat_idx].devices().size(); gpu_idx++){
        
            Rcout << "context id" << std::endl;
            Rcout << id << std::endl;
            
            Rcout << "current platform index" << std::endl;
            Rcout << plat_idx << std::endl;
            
            // Select the platform
            viennacl::ocl::switch_context(id);
            
            context_index[id] = id + 1;
            platform_index[id] = plat_idx;
            platform_name[id] = platforms[plat_idx].info();
            
            viennacl::ocl::set_context_platform_index(id, plat_idx);
            
            // Select device
            viennacl::ocl::get_context(id).switch_device(gpu_idx);
            
            // Get device info
            device_index[id] = gpu_idx;
            device_name[id] = viennacl::ocl::current_device().name();
            
            Rcout << "current device index" << std::endl;
            Rcout << device_index[id] << std::endl;
            
            Rcout << "current device name" << std::endl;
            Rcout << device_name[id] << std::endl;
            
            switch(viennacl::ocl::current_device().type()){
                case 2: 
                    device_type[id] = "cpu";
                    break;
                case 4: 
                    device_type[id] = "gpu";
                    break;
                case 8: 
                    device_type[id] = "accelerator";
                    break;
                default:
                    Rcpp::Rcout << "device found" << std::endl;
                    Rcpp::Rcout << viennacl::ocl::current_device().type() << std::endl;
                    throw Rcpp::exception("unrecognized device detected");
            }
        
            // increment context
            id++;
        }
    }
    
    viennacl::ocl::switch_context(current_context_id);
    
    return Rcpp::DataFrame::create(Rcpp::Named("context") = context_index,
    			  Rcpp::Named("platform") = platform_name,
                  Rcpp::Named("platform_index") = platform_index,
				  Rcpp::Named("device") = device_name,
                  Rcpp::Named("device_index") = device_index,
                  Rcpp::Named("device_type") = device_type);
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


//' @title Set Context
//' @description Change the current context used by default
//' @param id Integer identifying which context to set
//' @seealso \link{listContexts}
//' @export
// [[Rcpp::export]]
void
setContext(int id)
{
    if(id <= 0){
        stop("Index cannot be 0 or less");
    }
    viennacl::ocl::switch_context(id-1);
}


