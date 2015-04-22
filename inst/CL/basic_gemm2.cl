__kernel void iMatMult(const int Mdim, const int Ndim, const int Pdim, 
                       __global const int *A, __global const int *B, __global int *C) {
    
    int k,j;
    int i = get_global_id(0);
    int Awrk[1000];
    int tmp;

    for(k=0; k<Pdim; k++){
        Awrk[k] = A[i*Ndim+k];
    }

    for(j=0; j<Mdim; j++){
        tmp = 0;
        // Do the operation
        for(k=0; k < Pdim; k++){
            tmp += Awrk[k] * B[k*Pdim+j];
        }
        C[i*Ndim+j] += tmp;
    }
}
