

__kernel void update_kk(__global double *A, unsigned int N, unsigned int k)
{
    A[k * N + k] = sqrt(A[k * N + k]);
}

__kernel void update_k(__global double *A, const int upper, unsigned int N, unsigned int Npad, unsigned int k)
{
    int i = get_global_id(0);
    
    if(i > k && i < N) {
        double Akk = A[k * Npad + k];
        
        if(upper == 0){
            // lower
            A[i * Npad + k] = A[i * Npad + k] / Akk;
            
            // zero out the top too - only if in-place
            A[k * Npad + i] = 0;
        }else if(upper == 1){
            // upper???
            A[k * Npad + i] = A[k * Npad + i] / Akk;
            
            // zero out the top too - only if in-place
            A[i * Npad + k] = 0;
        }else{
            return;
        }
    }
}

__kernel void update_block(__global double *A, const int upper, unsigned int N, unsigned int Npad, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if(i <= k || j <= k) return;
    if(i >= N || j >  i) return;
    
    if(upper == 0){
        // lower
        double Aik = A[i * Npad + k];
        double Ajk = A[j * Npad + k];
        double Aij = A[i * Npad + j];
        
        A[i * Npad + j] = Aij - Aik * Ajk;
        
    }else if(upper ==1 ){
        // upper
        double Aik = A[k * Npad + i];
        double Ajk = A[k * Npad + j];
        double Aij = A[j * Npad + i];
        
        A[j * Npad + i] = Aij - Aik * Ajk;
    }else{
        return;
    }
}
