__kernel void iMatMult(const int Mdim, const int MdimPad, 
                       const int Pdim, const int PdimPad,
                       __global const int *A, __global const int *B, __global int *C) {
    
    // Get the index of the elements to be processed
    const int globalRow = get_global_id(0); // C Row ID
    const int globalCol = get_global_id(1); // C Col ID
    int tmp = 0;
    
    // Do the operation
    if((globalRow <= Mdim) && (globalCol <= Pdim)){
        
        for(int k=0; k < Pdim; k++){
            tmp += A[globalRow * MdimPad + k] * B[globalCol+PdimPad*k];
        }
        
        C[globalCol+MdimPad*globalRow] = tmp;
    }
}
