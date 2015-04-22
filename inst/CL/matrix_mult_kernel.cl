__kernel void iMatMult( const int N, 
                        __global int* A, 
                        __global int* B, 
                        __global int* C,
                        __local int* Bwrk) 
{
    int k, j;
    int i    = get_global_id(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);
    int Awrk[1024];
    int tmp;
    if (i < N) {
        for (k = 0; k < N; k++)
            Awrk[k] = A[i*N+k];

        for (j = 0; j < N; j++) {
            barrier(CLK_LOCAL_MEM_FENCE);
            for (k = iloc; k < N; k += nloc)
                Bwrk[k] = B[k*N+j];
            barrier(CLK_LOCAL_MEM_FENCE);
            tmp = 0.0f;
            for (k = 0; k < N; k++)
                tmp += Awrk[k] * Bwrk[k];
            C[i*N+j] = tmp;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}