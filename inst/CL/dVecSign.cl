__kernel void VecSign(
    __global double *A, __global double *B, const unsigned int n) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    
    // Do the operation
    if(globalRow < n){
        B[globalRow] = sign(A[globalRow]);
    }
}