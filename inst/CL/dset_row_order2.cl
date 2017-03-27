__kernel void set_row_order(
    __global const double *A, __global double *B, __global const int *indices,
    const int Mdim, const int globalCol, const int MdimPad) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    //const int globalCol = get_global_id(1); // C Col ID
    
    // Do the operation
    if(globalRow < Mdim){
        
        //printf("Mdim: %d\n", Mdim);
        //printf("globalRow: %d\n", globalRow);
    
        //printf("index: %d\n", indices[globalRow]);
        //printf("final index: %d\n", indices[globalRow] * MdimPad + globalCol);
        //printf("A index: %f\n", A[indices[globalRow] * MdimPad + globalCol]);
        
        B[globalRow] = A[indices[globalRow] * MdimPad + globalCol];
    }
}
