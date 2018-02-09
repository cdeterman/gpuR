AC_INIT([gpuR], 2.0.0)

AC_LANG(C++)
AC_REQUIRE_CPP

echo "Checking for C++ Compiler"
AC_PROG_CXX

GPUR_CPPFLAGS="-I. -I../inst/include"
GPUR_CXXFLAGS=""
GPUR_LIBS=""

if test "$BACKEND" = "OPENCL"; then

    AC_MSG_NOTICE("BACKEND = OpenCL")

    if test -z "{OPENCL_FLAGS}"; then
        AC_MSG_RESULT("OPENCL_FLAGS not set, using default -DCL_HPP_MINIMUM_OPENCL_VERSION=110 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS  -DCL_HPP_TARGET_OPENCL_VERSION=120")
        GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -DCL_HPP_MINIMUM_OPENCL_VERSION=110 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_HPP_TARGET_OPENCL_VERSION=120 -fPIC"
    else
        echo "Using OPENCL_FLAGS = " $OPENCL_FLAGS
        GPUR_CPPFLAGS="${OPENCL_FLAGS}"
    fi
    
    # check OS
    case `uname` in
        Darwin* )
            AC_MSG_NOTICE("Darwin OS")
    
            AC_MSG_CHECKING("whether OpenCL library available")
            
            if test -e "$OPENCL_LIB"; then
                AC_MSG_RESULT("OPENCL_LIB was previously set")
                # custom paths
                msg="Using OPENCL_LIB = " $OPENCL_LIB
                AC_MSG_RESULT([$msg])
                GPUR_LIBS="GPUR_LIBS=-L$OPENCL_LIB -framework OpenCL -Wl,-rpath,$OPENCL_LIB"
    
                if ${OPENCL_INC+"false"}; then
                    AC_MSG_RESULT([OPENCL_INC not set, using default include directory])
                    OPENCL_INC=/System/Library/Frameworks/OpenCL.framework/Headers/
                else
                    msg="Using OPENCL_INC = " $OPENCL_INC
                    AC_MSG_RESULT([$msg])
                fi
    
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$CUDA_HOME"; then
                # CUDA SDK
                AC_MSG_RESULT("CUDA_HOME previously set")
                OPENCL_INC=$CUDA_HOME/include
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                GPUR_LIBS="${GPUR_LIBS} -L$CUDA_HOME/lib64 -framework OpenCL -Wl,-rpath,$CUDA_HOME/lib64"
            elif test -e "$AMDAPPSDKROOT"; then
                # AMD SDK
                AC_MSG_RESULT("AMDAPPSDKROOT previously set")
                OPENCL_INC=$AMDAPPSDKROOT/include
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                GPUR_LIBS="${GPUR_LIBS} -L$AMDAPPSDKROOT/lib/x86_64 -framework OpenCL -Wl,-rpath,$AMDAPPSDKROOT/lib/x86_64"
            elif test -e "$INTELOCLSDKROOT"; then
                # Intel SDK
                AC_MSG_RESULT("INTELOCLSDKROOT previously set")
                echo "using INTELOCLSDKROOT"
                OPENCL_INC=$INTELOCLSDKROOT/include
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                GPUR_LIBS="${GPUR_LIBS} -L$INTELOCLSDKROOT/lib/x64 -framework -Wl,-rpath,$INTELOCLSDKROOT/lib/x64"
            else
                AC_MSG_RESULT("OPENCL_LIB not set, trying default lib location with -framework OpenCL")
                GPUR_LIBS="GPUR_LIBS=-framework OpenCL"
                
                AC_MSG_NOTICE("Checking OpenCL C++ API")
    
                if ${OPENCL_INC+"false"}; then
                    AC_MSG_RESULT("OPENCL_INC not set, using default include directory")
                    OPENCL_INC=/System/Library/Frameworks/OpenCL.framework/Headers/
                else
                    echo "Using OPENCL_INC = " $OPENCL_INC
                fi
            fi
    
            if test -e "$OPENCL_INC/cl2.hpp"; then
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I$OPENCL_INC -DHAVE_CL_CL2_HPP"
                AC_MSG_RESULT("cl2.hpp header exists")
            else
                if test -e "$OPENCL_INC/cl.hpp"; then
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I$OPENCL_INC"
                    AC_MSG_RESULT("cl.hpp header exists")
                fi
            fi
            ;;
        Linux* )
            AC_MSG_NOTICE("Linux OS")
            AC_MSG_CHECKING("whether OpenCL library available")
    
            if test -e "$OPENCL_LIB"; then
                AC_MSG_RESULT("OPENCL_LIB was previously set")
                # custom paths
                echo "Using OPENCL_LIB =" $OPENCL_LIB
                GPUR_LIBS="${GPUR_LIBS} -L$OPENCL_LIB -lOpenCL -Wl,-rpath,$OPENCL_LIB"
                if ${OPENCL_INC+"false"}; then
                    AC_MSG_RESULT("OPENCL_INC not set, using default include directory /usr/include")
                    OPENCL_INC=/usr/include
                else
                    echo "Using OPENCL_INC = " $OPENCL_INC
                fi
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$CUDA_HOME"; then
                # CUDA SDK
                AC_MSG_RESULT("CUDA_HOME previously set")
                OPENCL_INC=$CUDA_HOME/include
                GPUR_LIBS="GPUR_LIBS=-L$CUDA_HOME/lib64 -lOpenCL -Wl,-rpath,$CUDA_HOME/lib64"
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$AMDAPPSDKROOT"; then
                # AMD SDK
                AC_MSG_RESULT("AMDAPPSDKROOT previously set")
                OPENCL_INC=$AMDAPPSDKROOT/include
                GPUR_LIBS="${GPUR_LIBS} -L$AMDAPPSDKROOT/lib/x86_64 -lOpenCL -Wl,-rpath,$AMDAPPSDKROOT/lib/x86_64"
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$INTELOCLSDKROOT"; then
                # Intel SDK
                AC_MSG_RESULT("INTELOCLSDKROOT previously set")
                OPENCL_INC=$INTELOCLSDKROOT/include
                GPUR_LIBS="${GPUR_LIBS} -L$INTELOCLSDKROOT/lib/x64 -lOpenCL -Wl,-rpath,$INTELOCLSDKROOT/lib/x64"
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            else
                AC_MSG_NOTICE("Searching for OpenCL library")
                # try searching for opencl library
                # if fails, build library from scratch
                if test `ldconfig -p | grep 'libOpenCL\|lOpenCL' | wc -l` > 0; then
                    AC_MSG_RESULT("found OpenCL library")
                    GPUR_LIBS="${GPUR_LIBS} -lOpenCL"
                    
                    AC_MSG_NOTICE("Checking OpenCL C++ API")
        
                    if ${OPENCL_INC+"false"}; then
                        AC_MSG_RESULT("OPENCL_INC not set, using default include directory /usr/include")
                        OPENCL_INC=/usr/include/
                    else
                        echo "Using OPENCL_INC = " $OPENCL_INC
                    fi
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                else
                    AC_MSG_RESULT("no OpenCL found, OPENCL_LIB not set, will be compiled from source")
                    # exit 1
                    
                    # setting defaults to compile from source
                    GPUR_LIBS="${GPUR_LIBS} -L../inst/include/loader -lOpenCL -pthread"
                    BUILD_LIB="BUILD_LIB=../inst/include/loader/libOpenCL.a"
                    
                    AC_SUBST(BUILD_LIB)
        
                    AC_MSG_NOTICE("Checking OpenCL C++ API")
        
                    if ${OPENCL_INC+"false"}; then
                        AC_MSG_RESULT("OPENCL_INC not set, using default include directory /usr/include")
                        OPENCL_INC=/usr/include/
                    else
                        echo "Using OPENCL_INC = " $OPENCL_INC
                    fi
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                fi
                
            fi
    
            if test -e "$OPENCL_INC/CL/cl2.hpp"; then
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -DHAVE_CL_CL2_HPP"
                AC_MSG_RESULT("cl2.hpp header exists")
            else
                if test -e "$OPENCL_INC/CL/cl.hpp"; then
                    AC_MSG_RESULT("cl.hpp header exists")
                else
                    AC_MSG_RESULT("No OpenCL C++ API found, will use the headers contained in the package")
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I. -pthread"
                fi
            fi
            ;;
        Solaris*)
            AC_MSG_ERROR("SOLARIS currently not supported")
            ;;
        *)
            AC_MSG_ERROR("unknown: `uname`")
            ;;
    esac
    
    
elif test "$BACKEND" = "CUDA"; then

    AC_MSG_NOTICE("BACKEND = CUDA")
    AC_SUBST(BACKEND,["$BACKEND"])
    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -DBACKEND_CUDA"
    
    AC_MSG_CHECKING("Checking environment variable CUDA_HOME")
    if test -z "${CUDA_HOME}"; then
        CUDA_HOME=`find /usr/local/ -maxdepth 1 -type d -name "cuda*" | sort -V | tail -1`
        AC_MSG_RESULT("CUDA_HOME not set; using highest version found ${CUDA_HOME}")
    else
        AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
    fi
    
    AC_MSG_CHECKING("Parsing CUDA version")
    CUDA_VER=$(find /usr/local/ -maxdepth 1 -type d -name "cuda*" | sort -V | tail -1 | sed -r "s/[[a-z]]*|-||\///g")
    AC_MSG_RESULT("CUDA version being used is ${CUDA_VER}")
    
    #echo ${CUDA_VER} 9.1 | awk '{if ($1 < $2) print false; else print true}'
    CUDA_EXCEPTION=`echo ${CUDA_VER} 9.1 | awk '{if ($1 < $2) print "false"; else print "true"}'`
    echo ${CUDA_EXCEPTION}
    if [ "$CUDA_EXCEPTION" = true ]; then
        AC_MSG_NOTICE("Need special include")
        AC_SUBST(CRT_INCL,["$CUDA_HOME/include/crt"])
    else
        AC_MSG_NOTICE("All is good")
        AC_SUBST(CRT_INCL,"")
    fi
    
    AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
    if test -z "${HAS_NVCC}"; then
        AC_MSG_ERROR("NVCC not found; check CUDA install")
    else
        AC_MSG_RESULT("NVCC found")
    fi
    
    AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
    AC_CHECK_FILE([${CUDA_HOME}/lib64/libcudart.so], [BIT64="yes"])
    if test -z "${BIT64}"; then
        AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
        cu_libdir="${CUDA_HOME}/lib"
        #AC_SUBST(CUDA_LIBDIR,["$cu_libdir"])
    else
        AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
        cu_libdir="${CUDA_HOME}/lib64"
        #AC_SUBST(CUDA_LIBDIR,["$cu_libdir"])
    fi
    
    GPUR_LIBS="${GPUR_LIBS} -L${cu_libdir} -Wl,-rpath,${cu_libdir} -lcudart"
    
    
    ## look for Rscript, but use the one found via R_HOME to allow for multiple installations
    AC_DEFUN([AC_PROG_RSCRIPT], [AC_CHECK_PROG(RSCRIPT,Rscript,yes)])
    AC_PROG_RSCRIPT
    
    if test x"${RSCRIPT}" == x"yes" ; then
    
        R_HOME=`Rscript -e 'cat(R.home())'`
        
        rcpp_incl=`${R_HOME}/bin/Rscript -e 'Rcpp:::CxxFlags()'`
        rcpp_eigen_incl=`${R_HOME}/bin/Rscript -e "RcppEigen:::CxxFlags()"`
        vcl_incl=`${R_HOME}/bin/Rscript -e 'cat(paste("-I", system.file("include", package = "RViennaCL"), sep = ""))'`
        bh_incl=`${R_HOME}/bin/Rscript -e 'cat(paste("-I", system.file("include", package = "BH"), sep = ""))'`
        
        AC_SUBST(RCPP_INCL,["$rcpp_incl"])
        AC_SUBST(RCPPEIGEN_INCL,["$rcpp_eigen_incl"])
        AC_SUBST(VCL_INCL,["$vcl_incl"])
        AC_SUBST(BH_INCL, ["$bh_incl"])
    else
        AC_MSG_ERROR("
        Your installation does not appear to have Rscript installed.
        
        Please make sure that you have a working and complete R installation.
        ")
    #exit 1
    fi
    
    AC_MSG_CHECKING("building the nvcc command line")
    NVCC="${CUDA_HOME}/bin/nvcc"

    AC_SUBST(NVCC)
    
else

    AC_MSG_NOTICE("Default: BACKEND = OpenCL")

    if test -e "$OPENCL_FLAGS"; then
        echo "Using OPENCL_FLAGS = " $OPENCL_FLAGS
        GPUR_CPPFLAGS="${OPENCL_FLAGS}"
    else
        AC_MSG_RESULT("OPENCL_FLAGS not set, using default -DCL_HPP_MINIMUM_OPENCL_VERSION=110 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS  -DCL_HPP_TARGET_OPENCL_VERSION=120")
        GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -DCL_HPP_MINIMUM_OPENCL_VERSION=110 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_HPP_TARGET_OPENCL_VERSION=120 -fPIC"
    fi
    
    # check OS
    case `uname` in
        Darwin* )
            AC_MSG_NOTICE("Darwin OS")
    
            AC_MSG_CHECKING("whether OpenCL library available")
            
            if test -e "$OPENCL_LIB"; then
                AC_MSG_RESULT("OPENCL_LIB was previously set")
                # custom paths
                echo "Using OPENCL_LIB = " $OPENCL_LIB
                GPUR_LIBS="GPUR_LIBS=-L$OPENCL_LIB -framework OpenCL -Wl,-rpath,$OPENCL_LIB"
    
                if ${OPENCL_INC+"false"}; then
                    AC_MSG_RESULT("OPENCL_INC not set, using default include directory")
                    OPENCL_INC=/System/Library/Frameworks/OpenCL.framework/Headers/
                else
                    echo "Using OPENCL_INC = " $OPENCL_INC
                fi
    
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$CUDA_HOME"; then
                # CUDA SDK
                AC_MSG_RESULT("CUDA_HOME previously set")
                OPENCL_INC=$CUDA_HOME/include
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                GPUR_LIBS="${GPUR_LIBS} -L$CUDA_HOME/lib64 -framework OpenCL -Wl,-rpath,$CUDA_HOME/lib64"
            elif test -e "$AMDAPPSDKROOT"; then
                # AMD SDK
                AC_MSG_RESULT("AMDAPPSDKROOT previously set")
                OPENCL_INC=$AMDAPPSDKROOT/include
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                GPUR_LIBS="${GPUR_LIBS} -L$AMDAPPSDKROOT/lib/x86_64 -framework OpenCL -Wl,-rpath,$AMDAPPSDKROOT/lib/x86_64"
            elif test -e "$INTELOCLSDKROOT"; then
                # Intel SDK
                AC_MSG_RESULT("INTELOCLSDKROOT previously set")
                echo "using INTELOCLSDKROOT"
                OPENCL_INC=$INTELOCLSDKROOT/include
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                GPUR_LIBS="${GPUR_LIBS} -L$INTELOCLSDKROOT/lib/x64 -framework -Wl,-rpath,$INTELOCLSDKROOT/lib/x64"
            else
                AC_MSG_RESULT("OPENCL_LIB not set, trying default lib location with -framework OpenCL")
                GPUR_LIBS="GPUR_LIBS=-framework OpenCL"
                
                AC_MSG_NOTICE("Checking OpenCL C++ API")
    
                if ${OPENCL_INC+"false"}; then
                    AC_MSG_RESULT("OPENCL_INC not set, using default include directory")
                    OPENCL_INC=/System/Library/Frameworks/OpenCL.framework/Headers/
                else
                    echo "Using OPENCL_INC = " $OPENCL_INC
                fi
            fi
    
            if test -e "$OPENCL_INC/cl2.hpp"; then
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I$OPENCL_INC -DHAVE_CL_CL2_HPP"
                AC_MSG_RESULT("cl2.hpp header exists")
            else
                if test -e "$OPENCL_INC/cl.hpp"; then
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I$OPENCL_INC"
                    AC_MSG_RESULT("cl.hpp header exists")
                fi
            fi
            ;;
        Linux* )
            AC_MSG_NOTICE("Linux OS")
            AC_MSG_CHECKING("whether OpenCL library available")
    
            if test -e "$OPENCL_LIB"; then
                AC_MSG_RESULT("OPENCL_LIB was previously set")
                # custom paths
                echo "Using OPENCL_LIB =" $OPENCL_LIB
                GPUR_LIBS="${GPUR_LIBS} -L$OPENCL_LIB -lOpenCL -Wl,-rpath,$OPENCL_LIB"
                if ${OPENCL_INC+"false"}; then
                    AC_MSG_RESULT("OPENCL_INC not set, using default include directory /usr/include")
                    OPENCL_INC=/usr/include
                else
                    echo "Using OPENCL_INC = " $OPENCL_INC
                fi
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$CUDA_HOME"; then
                # CUDA SDK
                AC_MSG_RESULT("CUDA_HOME previously set")
                OPENCL_INC=$CUDA_HOME/include
                GPUR_LIBS="GPUR_LIBS=-L$CUDA_HOME/lib64 -lOpenCL -Wl,-rpath,$CUDA_HOME/lib64"
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$AMDAPPSDKROOT"; then
                # AMD SDK
                AC_MSG_RESULT("AMDAPPSDKROOT previously set")
                OPENCL_INC=$AMDAPPSDKROOT/include
                GPUR_LIBS="${GPUR_LIBS} -L$AMDAPPSDKROOT/lib/x86_64 -lOpenCL -Wl,-rpath,$AMDAPPSDKROOT/lib/x86_64"
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            elif test -e "$INTELOCLSDKROOT"; then
                # Intel SDK
                AC_MSG_RESULT("INTELOCLSDKROOT previously set")
                OPENCL_INC=$INTELOCLSDKROOT/include
                GPUR_LIBS="${GPUR_LIBS} -L$INTELOCLSDKROOT/lib/x64 -lOpenCL -Wl,-rpath,$INTELOCLSDKROOT/lib/x64"
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
            else
                AC_MSG_NOTICE("Searching for OpenCL library")
                # try searching for opencl library
                # if fails, build library from scratch
                if test `ldconfig -p | grep 'libOpenCL\|lOpenCL' | wc -l` > 0; then
                    AC_MSG_RESULT("found OpenCL library")
                    
                    AC_MSG_NOTICE("Pulling last OpenCL library listed")
                    ocl_lib_path=`ldconfig -p | grep 'libOpenCL.so$\|lOpenCL.so$' | cut -d ' ' -f4 | tail -n 1 | awk -F"libOpenCL" '{print $1}'`
                    
                    #GPUR_LIBS="${GPUR_LIBS} -lOpenCL"
                    GPUR_LIBS="${GPUR_LIBS} -L$ocl_lib_path -lOpenCL -Wl,-rpath,$ocl_lib_path"
    
                    AC_MSG_NOTICE("Checking OpenCL C++ API")
        
                    if ${OPENCL_INC+"false"}; then
                        AC_MSG_RESULT("OPENCL_INC not set, using default include directory /usr/include")
                        OPENCL_INC=/usr/include/
                    else
                        echo "Using OPENCL_INC = " $OPENCL_INC
                    fi
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
    
                else
                    AC_MSG_RESULT("no OpenCL found, OPENCL_LIB not set, will be compiled from source")
                    # exit 1
                    
                    # setting defaults to compile from source
                    GPUR_LIBS="${GPUR_LIBS} -L../inst/include/loader -lOpenCL -pthread"
                    BUILD_LIB="BUILD_LIB=../inst/include/loader/libOpenCL.a"
                    
                    AC_SUBST(BUILD_LIB)
        
                    AC_MSG_NOTICE("Checking OpenCL C++ API")
        
                    if ${OPENCL_INC+"false"}; then
                        AC_MSG_RESULT("OPENCL_INC not set, using default include directory /usr/include")
                        OPENCL_INC=/usr/include/
                    else
                        echo "Using OPENCL_INC = " $OPENCL_INC
                    fi
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I'${OPENCL_INC}'"
                fi
                
            fi
    
            if test -e "$OPENCL_INC/CL/cl2.hpp"; then
                GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -DHAVE_CL_CL2_HPP"
                AC_MSG_RESULT("cl2.hpp header exists")
            else
                if test -e "$OPENCL_INC/CL/cl.hpp"; then
                    AC_MSG_RESULT("cl.hpp header exists")
                else
                    AC_MSG_RESULT("No OpenCL C++ API found, will use the headers contained in the package")
                    GPUR_CPPFLAGS="${GPUR_CPPFLAGS} -I. -pthread"
                fi
            fi
    
            ;;
        Solaris*)
            AC_MSG_ERROR("SOLARIS currently not supported")
            ;;
        *)
            AC_MSG_ERROR("unknown: `uname`")
            ;;
    esac
    
fi


AC_MSG_NOTICE("Acquiring R compiler flags")
R_INCL=`R CMD config --cppflags`
R_LIBS=`R CMD config --ldflags`
R_CPIC=`R CMD config CPICFLAGS`



AC_MSG_NOTICE([Building Makevars])
AC_SUBST(GPUR_CPPFLAGS)
AC_SUBST(GPUR_CXXFLAGS)
AC_SUBST(R_INCL)
AC_SUBST(R_CPIC)
AC_SUBST(R_LIBS)
AC_SUBST(GPUR_LIBS)
AC_OUTPUT(src/Makevars)
