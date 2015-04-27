__kernel void iaxpy(__const int ALPHA, __global const int *A, __global int *B) {
        
    // Get the index of the elements to be processed
    const unsigned int i = get_global_id(0);

    // Do the operation
    B[i] = ALPHA * A[i] + B[i];
}
