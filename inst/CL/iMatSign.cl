__kernel void MatSign(
    __global const int *A, __global int *B,
    const int Mdim, const int Pdim, const int MdimPad) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    
    // Do the operation
    if((globalRow <= Mdim) && (globalCol <= Pdim)){
        
        B[globalRow * MdimPad + globalCol] = sign(A[globalRow * MdimPad + globalCol]);
    }
}
