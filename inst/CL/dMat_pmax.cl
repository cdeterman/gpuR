__kernel void pmax(
    __global const double *A, __global double *B, const double x,
    const int Mdim, const int Pdim, const int MdimPad) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    
    // Do the operation
    if((globalRow <= Mdim) && (globalCol <= Pdim)){
        
        B[globalRow * MdimPad + globalCol] = max(A[globalRow * MdimPad + globalCol], x);
    }
}