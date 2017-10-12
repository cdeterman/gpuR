__kernel void pmax(
    __global float *A, __global float *B, const float x, const unsigned int n) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    
    // Do the operation
    if(globalRow < n){
        B[globalRow] = min(A[globalRow], x);
    }
}