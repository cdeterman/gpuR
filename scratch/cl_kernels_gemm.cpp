__kernel void iMatMult( const int N, 
                        __global int* A, 
                        __global int* B, 
                        __global int* C) 
{ 
    int k; 
    int i = get_global_id(0); 
    int j = get_global_id(1); 
    
    int tmp; 
    if ( (i < N) && (j <N)) { 
        tmp = 0;
        
        for(k=0; k<N; k++) {
            tmp += A[i*N+k] * B[k*N+j]; 
        }
        C[i*N+j] = tmp; 
    } 
} 


__kernel void iMatMult( const int N, 
                        __global int* A, 
                        __global int* B, 
                        __global int* C) 
{
    int k, j;
    int i = get_global_id(0);
    int Awrk[1024];
    int tmp;
    if (i < N) {
        for (k = 0; k < N; k++)
            Awrk[k] = A[i*N+k];
        
        for (j = 0; j < N; j++) {
            tmp = 0.0f;
            for (k = 0; k < N; k++)
                tmp += Awrk[k] * B[k*N+j];
            C[i*N+j] = tmp;
        }
    }
}