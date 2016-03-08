AC_INIT([gpuR],[1.0.0])
AC_CONFIG_AUX_DIR([tools])

AC_LANG(C++)

AC_MSG_CHECKING("Checking for C++ Compiler")
AC_REQUIRE_CPP

AC_CANONICAL_HOST

case $host_os in
    darwin* )
        echo "Darwin OS"
        if ${OPENCL_LIB+"false"}; then
            echo "OPENCL_LIB not set, trying default lib location"
            PKG_LIBS="-framework OpenCL"
        else
            echo "Using OPENCL_LIB = " $OPENCL_LIB
            PKG_LIBS="-L$OPENCL_LIB -framework OpenCL -Wl,-rpath,$OPENCL_LIB"
        fi
        
        AC_MSG_CHECKING("Checking OpenCL C++ API")

        if ${OPENCL_INC+"false"}; then
            echo ""
            echo "OPENCL_INC not set, using default include directory"
            OPENCL_INC=/System/Library/Frameworks/OpenCL.framework/Headers/
        else
            echo ""
            echo "Using OPENCL_INC = " $OPENCL_INC
        fi

        if test -e "$OPENCL_INC/CL/cl2.hpp"; then
            D_CL_HPP=-DHAVE_OPENCL_CL2_HPP
            echo "cl2.hpp header exists"
        else
            if test -e "$OPENCL_INC/CL/cl.hpp"; then
                echo "cl.hpp header exists"
            else
                AC_MSG_ERROR("No OpenCL C++ API found; try setting OPENCL_INC")
            fi
        fi
        ;;
    linux* )
        echo "Linux OS"

        AC_MSG_CHECKING("Checking for OpenCL library path")
        if ${OPENCL_LIB+"false"}; then
            echo ""
            echo "OPENCL_LIB not set, trying default lib location"
            PKG_LIBS=-lOpenCL
        else
            echo ""
            echo "Using OPENCL_LIB = " $OPENCL_LIB
            PKG_LIBS="-L$OPENCL_LIB -lOpenCL -Wl,-rpath,$OPENCL_LIB"
        fi
        
        AC_MSG_CHECKING("Checking OpenCL C++ API")
        
        if ${OPENCL_INC+"false"}; then
            echo ""
            echo "OPENCL_INC not set, using default include directory /usr/include"
            OPENCL_INC=/usr/include/
        else
            echo ""
            echo "Using OPENCL_INC = " $OPENCL_INC
        fi

        if test -e "$OPENCL_INC/CL/cl2.hpp"; then
            D_CL_HPP=-DHAVE_CL_CL2_HPP
            echo "cl2.hpp header exists"
        else
            if test -e "$OPENCL_INC/CL/cl.hpp"; then
                echo "cl.hpp header exists"
            else
                AC_MSG_ERROR("No OpenCL C++ API found; try installing opencl-headers or setting OPENCL_INC")
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
