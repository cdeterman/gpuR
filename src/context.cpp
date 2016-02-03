
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


//' @export
// [[Rcpp::export]]
DataFrame 
listContexts()
{
    // declarations
    int id = 0;
    long current_context_id = viennacl::ocl::backend<>::current_context_id();
    int num_contexts = 0;
    int num_devices;
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    
    // get platforms
    platforms_type platforms = viennacl::ocl::get_platforms();  
    
    // count number of contexts initialized
    // for each platform    
    for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
        
        num_contexts += platforms[plat_idx].devices().size();
//        for(unsigned int gpu_idx=0; gpu_idx < platforms[plat_idx].devices().size(); gpu_idx++){
//            num_contexts++;
//        }
    }
    
    std::cout << num_contexts << std::endl;
    
    Rcpp::IntegerVector context_index(num_contexts);
    Rcpp::CharacterVector platform_name(num_contexts);
    Rcpp::IntegerVector platform_index(num_contexts);
    Rcpp::CharacterVector device_name(num_contexts);
    Rcpp::IntegerVector device_index(num_contexts);
    Rcpp::CharacterVector device_type(num_contexts);
    
    
    for(unsigned int plat_idx=0; plat_idx < platforms.size(); plat_idx++){
        
        for(unsigned int gpu_idx=0; gpu_idx < platforms[plat_idx].devices().size(); gpu_idx++){
        
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
                default: throw Rcpp::exception("unrecognized device detected");
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


//' @export
// [[Rcpp::export]]
int currentContext()
{
    return viennacl::ocl::backend<>::current_context_id() + 1;
}


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


