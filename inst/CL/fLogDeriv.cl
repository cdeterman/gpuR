__kernel void logistic_deriv(
    __global float *A,
    const int Mdim, const int Pdim, const int MdimPad) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    
    // Do the operation
    if((globalRow <= Mdim) && (globalCol <= Pdim)){
        A[globalRow * MdimPad + globalCol] = A[globalRow * MdimPad + globalCol] * ( 1 - A[globalRow * MdimPad + globalCol] );
    }
}