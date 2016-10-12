__kernel void set_row_order(
    __global const float *A, __global float *B, __global const *indices,
    const int Mdim, const int Pdim, const int MdimPad) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    
    // Do the operation
    if((globalRow <= Mdim) && (globalCol <= Pdim)){
    
        printf("index = %i\n", indices[globalRow]);
        printf("globalRow = %i\n", globalRow);
        
        B[globalRow * MdimPad + globalCol] = A[indices[globalRow] * MdimPad + globalCol];
    }
}
