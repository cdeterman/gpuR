//------------------------------------------------------------------------------
//
// Function to initialize the input matrices A and B
//
//------------------------------------------------------------------------------
void initmat(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    int i, j;
    /* Initialize matrices */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            A[i*N+j] = 3;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            B[i*N+j] = 5;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            C[i*N+j] = 0;
}