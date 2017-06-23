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


// Function to round down a valid to nearest 'multiple' (e.g. 16)
inline
int 
roundDown(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;
    
    int remainder = numToRound % multiple;
    if (remainder == 0 || remainder == numToRound)
        return numToRound;
    
    return numToRound - remainder;
}

inline
int 
roundUp(int numToRound, int multiple)
{
	if (multiple == 0)
		return numToRound;
	
	int remainder = numToRound % multiple;
	if (remainder == 0 || multiple == numToRound)
		return numToRound;
	
	return numToRound + multiple - remainder;
}

#endif
