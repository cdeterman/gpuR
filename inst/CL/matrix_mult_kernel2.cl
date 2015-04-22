#define BLOCK_SIZE 16

__kernel void iMatMult(__global int* C, 
                      __global int* A, 
                      __global int* B, int wA, int wB)
{

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = wA * BLOCK_SIZE * by;
 
    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + wA - 1;
 
    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = BLOCK_SIZE * bx;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;
 
    int Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
    {

        //int Csub = 0;

        // Declaration of the local memory array As 
        // used to store the sub-matrix of A
        __local int As[BLOCK_SIZE][BLOCK_SIZE];
 
        // Declaration of the local memory array Bs 
        // used to store the sub-matrix of B
        __local int Bs[BLOCK_SIZE][BLOCK_SIZE];
 
        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
 
        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}