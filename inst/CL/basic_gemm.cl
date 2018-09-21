__kernel void iMatMult(const int A_size1, 
                       const int A_internal_size1,
                       const int B_size1,
                       const int B_size2, 
                       const int B_internal_size2,
                       const int C_internal_size2, 
                       __global const int *A, __global const int *B, __global int *C) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    int tmp = 0;
    
    // Do the operation
    if((globalRow < A_size1) && (globalCol < B_size2)){
    //if((globalRow == 1) && (globalCol < 1)){
        
        // for each row in B column
        // multiply the A element
        for(int k=0; k < B_size1; k++){
            
            tmp += A[globalRow * A_internal_size1 + k] * B[globalCol+B_internal_size2*k];
        }
        
        C[globalCol+C_internal_size2*globalRow] = tmp;
    }
}
