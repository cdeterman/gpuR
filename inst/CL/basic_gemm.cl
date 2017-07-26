__kernel void iMatMult(const int A_size2, const int C_size2, 
                       const int B_size1, const int C_size1,
                       __global const int *A, __global const int *B, __global int *C) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    int tmp = 0;
    
    // Do the operation
    if((globalRow <= A_size2) && (globalCol <= B_size1)){
        
        for(int k=0; k < B_size1; k++){
            tmp += A[globalRow * C_size2 + k] * B[globalCol+C_size1*k];
        }
        
        C[globalCol+C_size2*globalRow] = tmp;
    }
}
