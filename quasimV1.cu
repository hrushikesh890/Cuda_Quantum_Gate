#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

float *fetch_data(float u[4], char* filename, int &qno, int &vec_length)
{
    FILE *file;
    bool bfirst = true;
    file = fopen(filename, "r");
    int i, j;
    char c;
    float rvar;
    float *local;
    float *vec;
    if (file == NULL)
    {
        printf("File Not Found!");
        return vec;
    }
    
    // Read the U matrix
    fscanf(file, "%f %f", &u[0], &u[1]);
    fscanf(file, "%f %f", &u[2], &u[3]);

    // Read the vector 
    i = 0;
    while (fscanf(file, "%f", &rvar) == 1)
    {
        if (bfirst)
        {
            local = (float *) malloc (sizeof(float));
            local[i] = rvar;
            i++;
            bfirst = false;
        }
        else
        {
            local = (float *) realloc (local, (i+1)*sizeof(float));
            local[i] = rvar;
            i++;
        }
    }
    
    qno = (int) local[i-1];
    vec = (float *) malloc ((i-1) * sizeof(float));
    vec_length = (i - 1);    

    for (j = 0; j < i-1; j++)
    {
        vec[j] = local[j];

    }
    
    free(local);
    return vec;
}

__global__ void
quantum_gate_multiply(int a, int b)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("%d %d i= %d\n", a, b, i);
}


int main(int argc, char *argv[])
{
    float u[4];
    
    int qno, vec_length;
    char* filename = argv[1];

    //printf("%s\n", filename);
    
    float *vec = fetch_data(u, filename, qno, vec_length);
    float *op = new float[vec_length];

    int mask = 0;
    int pw = (int)(log(vec_length)/log(2));
    int antimask = (int)(pow(2, (pw - 1)) - 1);
    
    for (int i = 0; i < qno; i++)
    {
        mask = ((mask << 1) | 1);
    }

    antimask = (antimask & (~mask));

    /*for (int j = 0; j < vec_length; j++)
    {
        printf("%f\n", vec[j]);
    }*/
    int threadsPerBlock = 256;
    int blocksPerGrid =(vec_length + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    //vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    //quantum_gate_multiply(u, vec, op, qno, vec_length, pw, mask, antimask);
    quantum_gate_multiply<<<blocksPerGrid, threadsPerBlock>>>(qno, vec_length);
    
    /*for (int j = 0; j < vec_length; j++)
    {
        printf("%.3f\n", op[j]);
    }*/

    free(vec);
    return 1;

}