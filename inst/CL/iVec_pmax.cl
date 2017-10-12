__kernel void pmax(
    __global int *A, __global int *B, const int x, const unsigned int n) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    
    // Do the operation
    if(globalRow < n){
        B[globalRow] = max(A[globalRow], x);
    }
}