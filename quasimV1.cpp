#include <stdio.h>
//#include <cuda_runtime.h>
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
    /*for (int j = 0; j < vec_length; j++)
    {
        printf("%f\n", vec[j]);
    }*/
    free(local);
    return vec;
}

void quantum_gate_multiply(float* u, float* vec, float* op, int qno, int vec_length, int pw, int mask, int antimask)
{
    /*for (int j = 0; j < vec_length; j++)
    {
        printf("%f\n", vec[j]);
    }
    printf("%f %f %f %f %d\n", u[0], u[1], u[2], u[3], vec_length);*/

    // int mask = 0;
    
    int idx1, idx2;

    

    for (int i = 0; i < (int)(pow(2, (pw - 1))); i++)
    {
        idx1 = (i & mask) | ((i & antimask) << 1);
        idx2 = (idx1 | (1 << qno));
        //printf("i - %d, %d, %d\n", i, idx1, idx2);
        op[idx1] = u[0] * vec[idx1] + u[1] * vec[idx2];
        op[idx2] = u[2] * vec[idx1] + u[3] * vec[idx2];
    }

    // printf("%d %d %f %d %d\n", pw, qno, u[3], antimask, mask);

    
}

int main(int argc, char *argv[]){
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

    quantum_gate_multiply(u, vec, op, qno, vec_length, pw, mask, antimask);
    
    for (int j = 0; j < vec_length; j++)
    {
        printf("%.3f\n", op[j]);
    }

    free(vec);
    return 1;

}