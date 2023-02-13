#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

float *fetch_data(float u[4], char* filename, int &qno, int &vec_length)
{
    FILE *file;
    bool bfirst = true;
    file = fopen(filename, "r");
    int i, j;
    //char c;
    float rvar;
    float *local;
    float *vec;
    
    if (file == NULL)
    {
        printf("File Not Found!");
        return 0;
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

    local[0] = vec[0];
    
    free(local);
    return vec;
}

__global__ void
quantum_gate_multiply(float* u, float* vec, float* op, int qno, int vec_length, int pw, int mask, int antimask)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int idx1, idx2;

    idx1 = (i & mask) | ((i & antimask) << 1);
    idx2 = (idx1 | (1 << qno));

    op[idx1] = u[0] * vec[idx1] + u[1] * vec[idx2];
    op[idx2] = u[2] * vec[idx1] + u[3] * vec[idx2];
    
    //printf("%d %d %d\n", i, idx1, idx2);
}


int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;
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

    // Code below taken from class vecAdd code
    size_t size = vec_length * sizeof(float);
    float *d_u = NULL;
    err = cudaMalloc((void **)&d_u, (4 * sizeof(float)));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector u (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_vec = NULL;
    err = cudaMalloc((void **)&d_vec, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector vec (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_op = NULL;
    err = cudaMalloc((void **)&d_op, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector op (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Cudamemcpy

    err = cudaMemcpy(d_u, u, (4 * sizeof(float)), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid =((vec_length/2) + threadsPerBlock - 1) / threadsPerBlock;
    
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    //vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    //quantum_gate_multiply(u, vec, op, qno, vec_length, pw, mask, antimask);

    struct timeval begin, end; 
    gettimeofday (&begin, NULL);
    
    quantum_gate_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_u, d_vec, d_op, qno, vec_length, pw, mask, antimask);
    
    gettimeofday (&end, NULL);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(op, d_op, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < vec_length; j++)
    {
        printf("%.3f\n", op[j]);
    }

    err = cudaFree(d_u);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_op);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_vec);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);
    printf("Time - %d\n", time_in_us);

    free(op);
    free(vec);
    return 1;

}