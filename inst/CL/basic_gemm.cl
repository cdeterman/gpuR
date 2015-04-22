__kernel void iMatMult(const int Mdim, const int Ndim, const int Pdim, 
                       __global const int *A, __global const int *B, __global int *C) {
    
    int k;
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    int tmp = 0;
    
    // Do the operation
    for(k=0; k < Pdim; k++){
        tmp += A[k*Mdim+globalRow] * B[globalCol*Pdim+k];
    }
    C[globalCol*Mdim+globalRow] = tmp;
}
