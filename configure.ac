AC_INIT([gpuR],[1.0.0])
AC_CONFIG_AUX_DIR([tools])

AC_LANG(C++)

AC_MSG_CHECKING("Checking for C++ Compiler")
AC_REQUIRE_CPP

AC_CANONICAL_HOST

case $host_os in
    darwin* )
        PKG_LIBS=-framework OpenCL
        echo "Darwin OS"
        CXX="clang++"
        CXXCPP="clang++ -E -std=gnu++11"
        CXXFLAGS="-g -O2 -std=gnu++11"
        
        AC_MSG_CHECKING("Checking OpenCL C++ API")
        AC_CHECK_HEADERS("OpenCL/cl2.hpp")
        AC_CHECK_HEADERS("OpenCL/cl.hpp")

        if test "$ac_cv_header_OpenCL_cl2_hpp" != no
        then
            D_CL_HPP=-DHAVE_OPENCL_CL2_HPP
        else
            if test "$ac_cv_header_OpenCL_cl_hpp" == no
            then
                AC_MSG_ERROR("No OpenCL C++ API; try installing opencl-headers")
            fi
        fi
        ;;
    linux* )
        PKG_LIBS=-lOpenCL
        echo "Linux OS"
        CXX="g++"
        CXXCPP="g++ -E -std=gnu++11"
        CXXFLAGS="-g -O2 -std=gnu++11"
        
        AC_MSG_CHECKING("Checking OpenCL C++ API")
        AC_CHECK_HEADERS("CL/cl2.hpp")
        AC_CHECK_HEADERS("CL/cl.hpp")
        
        if test "$ac_cv_header_CL_cl2_hpp" != no
        then
            D_CL_HPP=-DHAVE_CL_CL2_HPP
        else
            if test "$ac_cv_header_CL_cl_hpp" == no
            then
                AC_MSG_ERROR("No OpenCL C++ API; try installing opencl-headers")
            fi
        fi
        ;;
    *)
        #Default Case
        AC_MSG_ERROR([Your platform is not currently supported])
        ;;
esac

AC_MSG_NOTICE([Building Makevars])
AC_SUBST(PKG_LIBS)
AC_SUBST(D_CL_HPP)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
