__kernel void logistic_deriv(
    __global double *A, __global double *B) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    
    // Do the operation
    A[globalRow] = B[globalRow] * ( 1 - B[globalRow] );
}