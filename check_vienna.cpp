
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

#include <Rcpp.h>

//[[Rcpp::export]]
void check_vienna_device(){
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    platforms_type platforms = viennacl::ocl::get_platforms();
    
    bool is_first_element = true;
    for (platforms_type::iterator platform_iter  = platforms.begin();
                     platform_iter != platforms.end();
                   ++platform_iter)
    {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
    
    std::cout << "# =========================================" << std::endl;
    std::cout << "#         Platform Information             " << std::endl;
    std::cout << "# =========================================" << std::endl;
    
    std::cout << "#" << std::endl;
    std::cout << "# Vendor and version: " << platform_iter->info() << std::endl;
    std::cout << "#" << std::endl;
    
    if (is_first_element)
    {
    std::cout << "# ViennaCL uses this OpenCL platform by default." << std::endl;
    is_first_element = false;
     }
    
    
     std::cout << "# " << std::endl;
     std::cout << "# Available Devices: " << std::endl;
     std::cout << "# " << std::endl;
     for (devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
     {
       std::cout << std::endl;
    
       std::cout << "  -----------------------------------------" << std::endl;
       std::cout << iter->full_info();
       std::cout << "  -----------------------------------------" << std::endl;
     }
     std::cout << std::endl;
     std::cout << "###########################################" << std::endl;
     std::cout << std::endl;
    }
}
    