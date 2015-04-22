__kernel void iMatMult(const int Mdim, const int Ndim, const int Pdim, 
                       __global const int *A, __global const int *B, __global int *C) {
    
    int k, j;
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    //const int globalCol = get_global_id(1); // C Col ID
    int tmp;
    
    // Do the operation
    for(j=0; j < Ndim; j++){
        tmp = 0;
        for(k=0; k < Pdim; k++){
            tmp += A[globalRow*Mdim+k]*B[k*Pdim+j];
        }
        C[i*Mdim+j] += tmp;
    }
}


__kernel void iMatMult(const int Mdim, const int Ndim, const int Pdim, 
                       __global const int *A, __global const int *B, __global int *C) {
    
    int k, j;
    
    int i = get_global_id(0); // C Row ID
    
    int tmp;
    
    // Do the operation
    for(j=0; j < Mdim; j++){
        tmp = 0;
        for(k=0; k < Pdim; k++){
            tmp += A[i*Ndim+k]*B[k*Pdim+j];
        }
        C[i*Ndim+j] += tmp;
    }
}
