#pragma once
#ifndef WINDOWS_CHECK_HPP
#define WINDOWS_CHECK_HPP

// windows deal with clashes
#ifdef _WIN32
//#define R_NO_REMAP 1
#define STRICT_R_HEADERS 
//#include <stdlib.h>
//#include <R.h>
//#undef Realloc
//#define R_Realloc(p,n,t) (t *) R_chk_realloc( (void *)(p), (size_t)((n) * sizeof(t)) )
//#include <windows.h>
#endif

#endif