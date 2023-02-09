#include <stdio.h>
//#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

void fetch_data(float (&u)[2][2], float *vec, char* filename)
{
    FILE *file;
    file = fopen(filename, "r");
    int i, j;
    float rvar;
    if (file == NULL)
    {
        printf("File Not Found!");
        return;
    }
    
    // Read the U matrix
    fscanf(file, "%f %f", &u[0][0], &u[0][1]);
    fscanf(file, "%f %f", &u[1][0], &u[1][1]);

    // Read the vector 
    while (fscanf(file, "%f", &rvar) == 1)
    {
        
    }

    printf("%f %f %f %f\n", u[0][0], u[0][1], u[1][0], u[1][1]);
   
}

int main(int argc, char *argv[]){
    float u[2][2];
    float *vec;

    char* filename = argv[1];
    printf("%s\n", filename);
    printf("%f\n", u[0][0]);
    fetch_data(u, vec, filename);
    //u[0][1] += 1.33;
    printf("%f %f %f %f\n", u[0][0], u[0][1], u[1][0], u[1][1]);
    return 1;

}