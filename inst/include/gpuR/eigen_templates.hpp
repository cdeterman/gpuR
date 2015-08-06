#pragma once
#ifndef EIGEN_TEMPLATES
#define EIGEN_TEMPLATES

// Would very much prefer to use this new C++11 syntax
// but the Travis-CI g++ always defaults to use C++0x which is 
// failing with this alias typedef so using the hideous struct below
// requiring the terrible MapMat<T>::Type syntax which also requires
// a typename declaration in each instance

template<class T>
using MapMat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;


template<class T>
using MapVec = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >;

#endif
