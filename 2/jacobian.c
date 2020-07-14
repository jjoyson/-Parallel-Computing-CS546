#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

float **a;          // MATRIX A
float *b;           // VECTOR B
int n = 5;          // VECTOR SIZE
int iter = 100;     // ITERATIONS
double begin;       // BEGIN TIME

// CALCULATION TIMES
double reduce_openMP_cal;
double collapse_openMP_cal; 
double serial_cal;

// SETUP TIMES
double create_openMP_setup;
double reduce_openMP_setup;
double collapse_openMP_setup;
double serial_setup;

// MATRIX & VECTOR SETUP
void createMatrixAndVector();
void createMatrixAndVectorOpenMP();
void freeMatrix();

// PRINTING
void printMatrixAndVector();
void printResults();

// ALGORITHMS
void serial();
void collapseOpenMP();
void reduceOpenMP();

void main(int argc, char *argv[]){

    if (errno != 0 || argc != 3){
        printf("\nIncorrect arguments: Should be in the following format: ");
        printf("./ans n iter\n");
        printf("\nWhere n is an integer (size of square matrix) and iter is an integer (number of iteration)\n");
        printf("\nRunning Default n = 5 and iter = 100\n\n");
    }

    else{
        n = atoi(argv[1]);
        iter = atoi(argv[2]);
    }

    b = (float *) malloc(sizeof(float) * n);
    a = (float **) malloc(sizeof(float*) * n);

    printf("Do you want to use converging 3x3 matrix and vecor (Y/...): ");
    int test = getchar();
    
    // FLUSH BUFFER
    if(test != '\n'){
        while(getchar() != '\n'){
            test = test;
        }
    }

    // CONVERGING MATRIX TEST
    if((char)test == 'Y'){
        n = 3;
        int i;
        for(i = 0; i < n; i++){
            a[i] = (float *) malloc(n*sizeof(float));
        }
        a[0][0] = 6; a[0][1] = 2; a[0][2] = 1;
        a[1][0] = 4; a[1][1] = 10; a[1][2] = 2;
        a[2][0] = 3; a[2][1] = 4; a[2][2] =14;

        b[0] = 3; b[1] = 4; b[2] = 2;
        // b[0]=17;b[1]=-18;b[2]=25;
        // a[0][0]=20;a[1][1]=20;a[2][2]=20;
        // a[0][1]=20;a[0][2]=-2;
        // a[1][0]=3;a[1][2]=-1;
        // a[2][0]=2;a[2][1]=-3; 
    }
    else
        createMatrixAndVector();

    printf("Do you want to print A matrix and b vector? (Y/...): ");
    int c = getchar();
    
    // FLUSH BUFFER
    if(c != '\n'){
        while(getchar() != '\n'){
            c = c;
        }
    }

    if((char)c == 'Y')
        printMatrixAndVector();
    else
        printf("Skipping printing...\n\n");

    serial();
    
    if((char)test != 'Y'){
        freeMatrix();
        createMatrixAndVectorOpenMP();
    }

    reduceOpenMP();
    collapseOpenMP();

    printResults();
}

void printResults(){
    printf("\n-------------------------------------------\n\n");
    printf("RESULTS:\n\n");
    printf("%d iterations & %d vector size\n\n",iter,n);

    printf("Setup Time:\n\n");
    printf("Serial Jacobian: %f seconds\n",serial_setup);
    printf("OpenMP Reduction Jacobian: %f seconds\n",reduce_openMP_setup);
    printf("OpenMP Collapse Jacobian: %f seconds\n",collapse_openMP_setup);

    printf("\nCalculation Time:\n\n");
    printf("Serial Jacobian: %f seconds\n",serial_cal);
    printf("OpenMP Reduction Jacobian: %f seconds\n",reduce_openMP_cal);
    printf("OpenMP Collapse Jacobian: %f seconds\n",collapse_openMP_cal);

    printf("\nSpeedUp:\n\n");
    printf("OpenMP Reduction Jacobian: %f seconds\n",serial_cal/reduce_openMP_cal);
    printf("OpenMP Collapse Jacobian: %f seconds\n",serial_cal/collapse_openMP_cal);
    
    printf("\nEfficiency:\n\n");
    printf("OpenMP Reduction Jacobian: %f seconds\n",serial_cal/(reduce_openMP_cal * omp_get_max_threads()));
    printf("OpenMP Collapse Jacobian: %f seconds\n",serial_cal/(collapse_openMP_cal * omp_get_max_threads()));
}

void freeMatrix(){
    printf("Freeing Matrix...\n\n");
    int i;
    for(i = 0; i < n; i++){
        free(a[i]);
    }
}

void createMatrixAndVector(){
    printf("Creating Matrix & Vector via Serial...\n\n");
    int i, j;
    begin = omp_get_wtime();
    for(i = 0; i < n; i++){
        b[i] = i+1;
        a[i] = (float *) malloc(n*sizeof(float));
        for(j = 0; j < n; j++){
            a[i][j] = (i+1)*(j+1);
        }
    }
    serial_setup = omp_get_wtime() - begin;
}

void createMatrixAndVectorOpenMP(){
    printf("Creating Matrix & Vector via OpenMP...\n\n");
    int i, j;
    begin = omp_get_wtime();

    #pragma omp parallel for
    for(i = 0; i < n; i++){
        b[i] = i+1;
        a[i] = (float *) malloc(n*sizeof(float));
    }

    #pragma omp parallel for collapse(2)
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            a[i][j] = (i+1)*(j+1);
        }
    }

    create_openMP_setup = omp_get_wtime() - begin;
}

void printMatrixAndVector(){
    int i, j;
    printf("\nPrinting Matrix A:\n");
    for(i = 0; i < n; i++){
        printf("[ ");
        for(j = 0; j < n; j++){
            printf("%f",a[i][j]);
            if( j < n-1)
                printf(", ");
        }
        printf("]\n");
    }
    printf("\nPrinting Vector b:\n[");
    for(i = 0; i < n; i++){
        printf("%f",b[i]);
        if( i < n-1)
            printf(", ");
    }
    printf("]\n\n");
}

void serial(){
    printf("Running Serial Jacobian...\n\n");
    float temp[n], x[n];
    float summation;
    int i, j, k;

    begin = omp_get_wtime();

    for(i = 0; i < n; i++){
        x[i] = 0;                                           // Initialize x with 0s
    }

    serial_setup += (omp_get_wtime() - begin);;

    begin = omp_get_wtime();

    for(k = 0; k < iter; k++){                              // k Iterations
        for(i = 0; i < n; i++){                             // n elements
            summation = 0;                                  // Initialize Summation
            for(j = 0; j < n; j++){
                if(i != j){
                    summation += a[i][j]*x[j];              // Loop n add
                }
            }
            temp[i] = (b[i] - summation)/a[i][i];            // Store new x in temp
        }
        for(i = 0; i < n; i++){
            x[i] = temp[i];                                 // after n calaculations, restore new x in x
        }
    }
    serial_cal = omp_get_wtime() - begin;

    printf("Do you want to print the resulting x values? (Y/...): ");
    
    int c = getchar();
    
    // FLUSH BUFFER
    if(c != '\n'){
        while(getchar() != '\n'){
            c = c;
        }
    }

    if((char)c == 'Y'){
        printf("\n");
        for(i = 0; i < n; i++){
            printf("x[%d] = %f\n",i,x[i]);
        }
        printf("\n");
    }
    else
        printf("Skipping printing...\n");

    return;
}

void reduceOpenMP(){
    printf("Running Reduction OpenMP Jacobian...\n\n");
    float temp[n], x[n], d[n], summation;
    int i, j, k;

    begin = omp_get_wtime();

    #pragma omp parallel for
    for(i = 0; i < n; i++){
        x[i] = 0;
    }

    reduce_openMP_setup = create_openMP_setup + (omp_get_wtime() - begin);

    begin = omp_get_wtime();

    for(k = 0; k < iter; k++){
        for(i = 0; i < n; i++){
            summation = b[i];
            #pragma omp parallel for reduction (- : summation)                  // Reduce summation over parallel processes
            for(j = 0; j < n; j++){
                if(i != j)
                    summation -= a[i][j]*x[j];
            }
            temp[i] = summation/a[i][i];
        }
        
        #pragma omp parallel for
        for(i = 0; i < n; i++){
            x[i] = temp[i];
        }
    }

    reduce_openMP_cal = omp_get_wtime() - begin;

    printf("Do you want to print the resulting x values? (Y/...): ");
    
    int c = getchar();
    
    // FLUSH BUFFER
    if(c != '\n'){
        while(getchar() != '\n'){
            c = c;
        }
    }

    if((char)c == 'Y'){
        printf("\n");
        for(i = 0; i < n; i++){
            printf("x[%d] = %f\n",i,x[i]);
        }
        printf("\n");
    }
    else
        printf("Skipping printing...\n");

    return;
}

void collapseOpenMP(){
    printf("Running Collapse OpenMP Jacobian...\n\n");
    float temp[n], x[n];
    int i, j, k;

    begin = omp_get_wtime();

    #pragma omp parallel for
    for(i = 0; i < n; i++){
        x[i] = 0;
        temp[i] = b[i]/a[i][i];                                 // Initialize temp for perfect nested for loop
    }

    collapse_openMP_setup = create_openMP_setup + (omp_get_wtime() - begin);
    begin = omp_get_wtime();

    for(k = 0; k < iter; k++){

        #pragma omp parallel for collapse(2) private(i,j)       // colapse for perfect nested for loop
        for(i = 0; i < n; i++){
            for(j = 0; j < n; j++){
                if(i != j){
                    temp[i] -= (a[i][j]*x[j])/a[i][i];          // Changes to formula for perfect nested for loop
                }
            }
        }

        for(i = 0; i < n; i++){
            x[i] = temp[i];
            temp[i] = b[i]/a[i][i];
        }
    }

    collapse_openMP_cal = omp_get_wtime() - begin;

    printf("Do you want to print the resulting x values? (Y/...): ");
    
    int c = getchar();

    // FLUSH OUT BUFFER
    if(c != '\n'){
        while(getchar() != '\n'){
            c = c;
        }
    }

    if((char)c == 'Y'){
        printf("\n");
        for(i = 0; i < n; i++){
            printf("x[%d] = %f\n",i,x[i]);
        }
        printf("\n");
    }
    else
        printf("Skipping printing...\n");

    return;
}