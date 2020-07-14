#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void lu_decomp(int n, int result, float * A, float ** B);
void temp_lu_decomp(int n, float ** a, float * b);

void main(int argc, char *argv[]){

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n,m,i,j,k;
    double st1, st2, pt1, pt2;
    char ans;
    
    // INITIAL QUESTIONS
    if(rank == 0){
        printf("Do you want to use provided test case (Y/...)? ");
        scanf("%s", &ans);

        if(ans == 'Y'){
            printf("Enter the size of the matrix (1 value): ");
            scanf("%d", &n);
            ans = 'D';
        }

        else{
            printf("Do you want to use created test case 1 (Y/...)? ");
            scanf("%s", &ans);

            if(ans == 'Y'){
                printf("Enter the size of the matrix (1 value): ");
                scanf("%d", &n);
                ans = '1';
            }

            else{
                printf("Do you want to use created test case 2 (Y/...)? ");
                scanf("%s", &ans);

                if(ans == 'Y'){
                    printf("Enter the size of the matrix (1 value): ");
                    scanf("%d", &n);
                    ans = '2';
                }

                else{
                    printf("Please edit created test case 1 or 2 and rerun program!");
                    return;
                }
            }

        }

        for(i = 1; i < size; i++){
            MPI_Send(&n, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&ans, 1, MPI_CHAR, i, 2, MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Recv(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&ans, 1, MPI_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    m = n/size;

    // CHECKING M
    if(n%size != 0){
        printf("For conviniece assumption is m*p = n\n");
        printf("m = %d, p = %d and m*p != n = %d\n", m,size,n);
        printf("Please change the number of processors!\n");
        return;
    }

    float * A_new = malloc(sizeof(float) * m * 3);
    float * d = malloc(sizeof(float) * m);

// STEP 1
    if(rank == 0){
        float *A;
        float *B;
        float *b;
        if(ans == 'D'){
            A = malloc(sizeof(float) * n * 3);
            B = malloc(sizeof(float) * n);
            b = malloc(sizeof(float) * n);

            FILE *stream = fopen("Provided/a.csv", "r");
            for(i = 1; i < n; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3]);
            }
            A[0] = 0;
            stream = fopen("Provided/b.csv", "r");
            for(i = 0; i < n; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3+1]);
            }

            stream = fopen("Provided/c.csv", "r");
            for(i = -1; i < n-1; i++){
                if(i == -1)
                    fscanf(stream, "%f,[^\n]", &A[3*n-1]);
                else
                    fscanf(stream, "%f,[^\n]", &A[i*3+2]);
            }
            A[3*n-1] = 0;

            stream = fopen("Provided/d.csv", "r");
            for(i = 0; i < n; i++){
                fscanf(stream, "%f,[^\n]", &B[i]);
                b[i] = B[i];
            }
            /*
            printf("MATRIX ai, bi, ci, di\n");
            for(i = 0; i < n; i++){
                printf("%f, %f, %f, %f\n",A[i*3],A[i*3+1],A[i*3+2],B[i]);
            }
            */
        }
        if(ans == '1'){
            if(size != 2){
                printf("Test Case checking for correctness of p = 2. P = %d.\nPLease enter the right # of processors\n",size);
                return;
            }
            A = malloc(sizeof(float) * n * 3);
            B = malloc(sizeof(float) * n);
            b = malloc(sizeof(float) * n);

            FILE *stream = fopen("Correctness_1/a.csv", "r");
            for(i = 1; i < n; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3]);
            }
            A[0] = 0;

            stream = fopen("Correctness_1/b.csv", "r");
            for(i = 0; i < n; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3+1]);
            }

            stream = fopen("Correctness_1/c.csv", "r");
            for(i = 0; i < n-1; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3+2]);
            }
            A[3*n-1] = 0;

            stream = fopen("Correctness_1/d.csv", "r");
            for(i = 0; i < n; i++){
                fscanf(stream, "%f,[^\n]", &B[i]);
                b[i] = B[i];
            }
        }
        if(ans == '2'){
            if(size <= 2){
                printf("Test Case checking for correctness of p > 2. P = %d.\nPLease enter the right # of processors\n",size);
                return;
            }
            A = malloc(sizeof(float) * n * 3);
            B = malloc(sizeof(float) * n);
            b = malloc(sizeof(float) * n);

            FILE *stream = fopen("Correctness_2/a.csv", "r");
            for(i = 1; i < n; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3]);
            }
            A[0] = 0;

            stream = fopen("Correctness_2/b.csv", "r");
            for(i = 0; i < n; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3+1]);
            }

            stream = fopen("Correctness_2/c.csv", "r");
            for(i = 0; i < n-1; i++){
                fscanf(stream, "%f,[^\n]", &A[i*3+2]);
            }
            A[3*n-1] = 0;

            stream = fopen("Correctness_2/d.csv", "r");
            for(i = 0; i < n; i++){
                fscanf(stream, "%f,[^\n]", &B[i]);
                b[i] = B[i];
            }
        }

        // Serial LU Decomp

        float **temp = malloc(sizeof(float *));
        temp[0] = b;
        st1 = MPI_Wtime();
        lu_decomp(n, 1, A, temp);
        st2 = MPI_Wtime();
        
        if(ans != 'D'){
            printf("Serial Results:\n");
            for(i = 0; i < n; i++){
                printf("x[%d] = %0.3f\n",i,temp[0][i]);
            }
        }

        printf("Serial Execution Time %1.2f seconds\n",st2-st1);
        free(temp);
        free(b);

        pt1 = MPI_Wtime();
        for(i = 1; i < size; i++){
            MPI_Send(&A[i*m*3], m*3, MPI_FLOAT, i, i, MPI_COMM_WORLD);
            MPI_Send(&B[i*m], m, MPI_FLOAT, i, i+1, MPI_COMM_WORLD);
        }
        for(i = 0; i < m*3; i++){
            if(i < m)
                d[i] = B[i];
            A_new[i] = A[i];
        }
        free(A);
        free(B);
    }
    else{
        MPI_Recv(A_new, m*3, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(d, m, MPI_FLOAT, 0, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
    
    }
/*
    for(i = 0; i < size; i++){
        if(i == rank){
            printf("For Proc: %d\n",rank);
            for(j = 0; j < m*3; j++){
                printf("%f, %f\n",A_new[j],d[j]);
            }
        }
    }
*/
    float ** D = malloc(sizeof(float*) * 3);
    D[0] = d;

    float e_0[m];
    memset(e_0, 0, m*sizeof(float));

    e_0[0] = A_new[0]; 
    D[1] = e_0;

    float e_m[m];
    memset(e_m, 0, m*sizeof(float));

    e_m[m-1] = A_new[m*3-1];
    D[2] = e_m;

    // STEP 2
    lu_decomp(m, 3, A_new, D);
/*
    for(i=0; i<3; i++)
    {
        printf("X[%d] Proc: %d\n",i,rank);
        for(j=0; j<m; j++)
            printf("%9.3f",D[i][j]);
        printf("\n");
    }
*/
    // STEP 3
    float x_next = 0;
    float v_next = 0;
/*
    for(i = 0; i<2; i++){
        printf("Proc: %d, v[%d] = %9.3f and w[%d] = %9.3f and x[%d] = %9.3f\n",rank,i,v[rank][i],i,w[rank][i],i,x[rank][i]);
    }
*/
    
    if(rank != 0){
        MPI_Send(&D[0][0], 1, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
        MPI_Send(&D[1][0], 1, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD);
    }

    if(rank != size-1){
        MPI_Recv(&x_next, 1, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&v_next, 1, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
/*
    for(i = 0; i < size; i++){
        if(i == rank){
            printf("For Proc: %d\n",rank);
            for(j = 0; j < size; j++){
                for(k = 0; k < 2; k++){
                    printf("x: %f, v: %f, w:%f\n",x[j+k],v[j][k],w[j][k]);
                }
            }
        }
    }
*/

    // STEP 4
    float **z = malloc(sizeof(float*) * 2);
    z[0] = malloc(sizeof(float) * 2);
    z[1] = malloc(sizeof(float) * 2);
    z[0][0] = 1;
    z[0][1] = D[2][m-1];
    z[1][0] = v_next;
    z[1][1] = 1;

    float *h = malloc(sizeof(float) * 2);
    h[0] = D[0][m-1];
    h[1] = x_next;

    temp_lu_decomp(2, z, h);
/*
    free(z[0]);
    free(z[1]);
    free(z);
*/
    float y2i = 0;

    if(rank != size - 1)
        MPI_Send(&h[0], 1, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);

    if(rank != 0)
        MPI_Recv(&y2i, 1, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


/*
    for(i = 0; i < size; i++){
        if(i == rank){
            for(j=0; j<2*(size - 1); j++)
            {
                printf("z[%d]\n",j);
                for(k=0; k<2*(size - 1); k++)
                    printf("%9.3f",z[j][k]);
                printf("\n");
            }
        }
    }

    if(rank ==0){
        for(i = 0; i < 2*(size-1); i++){
            printf("%f, ",h[i]);
        }
        printf("\n");
    }*/


    // STEP 5
    float d_x[m];
    for(int i = 0; i < m; i++){
        d_x[i] = D[1][i]*y2i + D[2][i]*h[0];
        d[i] =  D[0][i] - d_x[i];
    }


    MPI_Barrier(MPI_COMM_WORLD);
    pt2 = MPI_Wtime();

     char *file_name;
    if(ans == 'D')
        file_name = "Provided/final_results.csv";
    if(ans == '1')
        file_name = "Correctness_1/final_results.csv";
    if(ans == '2')
        file_name = "Correctness_2/final_results.csv";
    FILE * results;
    char temp;

    if(rank == 0){
        if(ans != 'D')
            printf("\nParallel Results:\n");
        results = fopen(file_name, "w+");
        for(i = 0; i < m; i++){
            fprintf(results,"%0.3f\n", d[i]);
            if(ans != 'D')
                printf("x[%d] = %0.3f\n",rank*m+i,d[i]);
        }
        fclose(results);

        temp = 'Y';
        if(rank < size-1)
            MPI_Send(&temp, 1, MPI_CHAR, rank + 1, rank + 1, MPI_COMM_WORLD);
        
        free(D);
        free(h);
    }
    else{
        MPI_Recv(&temp, 1, MPI_CHAR, rank - 1, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        results = fopen(file_name, "a");
        for(i = 0; i < m; i++){
            fprintf(results,"%0.3f\n", d[i]);
            if(ans != 'D')
                printf("x[%d] = %0.3f\n",rank*m+i,d[i]);
        }
        fclose(results);

        if(rank < size-1)
            MPI_Send(&temp, 1, MPI_CHAR, rank + 1, rank + 1, MPI_COMM_WORLD);
        
        free(D);
        free(h);
    }

    MPI_Barrier(MPI_COMM_WORLD);  //Printing Time Purposes
    if(rank == 0)
        printf("Execution Time %1.2f seconds\n",pt2-pt1);
    
    MPI_Finalize();
    return;
}

void temp_lu_decomp(int n, float ** a, float * b){
    int i,j,k;
    float l[n][n];
    float u[n][n];

    for(i =0; i < n; i++){
        for(j =0; j < n; j++){
	        if(j<i)
                l[j][i] = 0;
            else {
                l[j][i] = a[j][i];
                for (k = 0; k < i; k++) {
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }
        }

        for (j = 0; j < n; j++) {
            if (j < i)
                u[i][j] = 0;
            else if (j == i)
                u[i][j] = 1;
            else {
                u[i][j] = a[i][j] / l[i][i];
                for (k = 0; k < i; k++) {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
    /*
    if(rank == 0){
        printf("L[%d]:\n",rank);
        for(i = 0; i < n; i++){
            printf("[");
            for(j = 0; j < n; j++){
                printf("%f ", l[i][j]);
            }
            printf("]\n");
        }
        printf("\n");

        printf("u[%d]:\n",rank);
        for(i = 0; i < n; i++){
            printf("[");
            for(j = 0; j < n; j++){
                printf("%f ", u[i][j]);
            }
            printf("]\n");
        }
        printf("\n");
    }
*/
    for(i=0; i<n; i++)
    {
        for(j=0; j<i; j++)
        {
            b[i]-=l[i][j]*b[j];
        }
        b[i]/=l[i][i];
    }
    for(i=n-1; i>=0; i--)
    {
        for(j=i+1; j<n; j++)
        {
            b[i]-=u[i][j]*b[j];
        }
    }
}

void lu_decomp(int n, int result, float * A, float ** D){
    int i,k;
    float L[n][2];
    float U[n][2];
/*
    for(i=0; i<result; i++)
    {
        printf("D[%d] Proc: %d\n",i,rank);
        for(k=0; k<n; k++)
            printf("%9.3f\n",D[i][k]);
        printf("\n");
    }
*/
    // LU Decomp
    
    for(i = 0; i < n; i++){
        L[i][1] = 1;
        if(i<1)
            U[i][1] = A[3*i+1];
        else{
            L[i][0] = A[i*3+0]/U[i-1][1];
            U[i][1] = A[i*3+1]-L[i][0]*A[(i-1)*3+2];
        }
        if( i > 0){
            for(k = 0; k<result; k++){
                D[k][i] = D[k][i] - L[i][0]*D[k][i-1];
            }
        }
    }
    
    for(i = n-1; i >= 0; i--){
        for(k = 0; k<result; k++){
            if(i > n-2)
                D[k][i] = (D[k][i])/U[i][1];
            else
                D[k][i] = (D[k][i] - A[i*3+2]*D[k][i+1])/U[i][1];
        }
    }
    return;
}

