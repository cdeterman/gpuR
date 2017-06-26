__kernel void pmax(
    __global double *A, __global double *B, const double x, const unsigned int n) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    
    // Do the operation
    if(globalRow < n){
        B[globalRow] = min(A[globalRow], x);
    }
}