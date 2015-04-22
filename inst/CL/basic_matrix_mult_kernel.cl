__kernel
void iMatMult(__global int* A, __global int* B, __global int* C,  int widthA, int widthB )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int value=0;
    for ( int k = 0; k < widthA; k++)
    {
        value += A[k + j * widthA] * B[k*widthB + i];
    }
    C[i + widthA * j] = value;
}