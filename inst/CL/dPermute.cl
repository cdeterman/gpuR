__kernel void dVecPermute(
    __global const double *A,
    __global double *B,
    __global const unsigned int *mask) {
    
    //*B = shuffle(*A, *mask);
    
    B[globalRow * MdimPad + globalCol] = A[indices[globalRow] * MdimPad + globalCol];
    
}
