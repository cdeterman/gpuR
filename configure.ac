AC_INIT([gpuR],[1.0.0])

AC_LANG(C++)
AC_REQUIRE_CPP

AC_MSG_CHECKING("Checking for C++ Compiler")

AX_CXX_COMPILE_STDCXX_11(,[mandatory])
CXXCPP="g++ -E -std=gnu++11"

AC_MSG_CHECKING("Checking OpenCL C++ API")
AC_CHECK_HEADERS("CL/cl2.hpp")

if test "$ac_cv_header_cl_cl2_hpp" != no
then
    D_CL_HPP=-DHAVE_CL_CL2_HPP
fi

AC_CANONICAL_HOST

case $host_os in
    darwin* )
        PKG_LIBS=-framework OpenCL
        echo "Darwin OS"
        ;;
    linux* )
        PKG_LIBS=-lOpenCL
        echo "Linux OS"
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
