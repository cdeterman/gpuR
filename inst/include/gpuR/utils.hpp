#ifndef GPUR_UTILS
#define GPUR_UTILS

#include <Rcpp.h>

inline
std::vector<std::string> 
split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

#endif
