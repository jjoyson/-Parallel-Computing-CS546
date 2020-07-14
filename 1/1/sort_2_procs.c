/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/*
 * Sort array using blocking send/recv between 2 ranks.
 *
 * The master process prepares the data and sends the latter half
 * of the array to the other rank. Each rank sorts it half. The
 * master then merges the sorted halves together. The two ranks
 * communicate using blocking send/recv.
 */

#define NUM_ELEMENTS 50

static int compare_int(const void *a, const void *b)
{
    return (*(int *) a - *(int *) b);
}

/* Merge sorted arrays a[] and b[] into a[].
 * Length of a[] must be sum of lengths of a[] and b[] */
static void merge(int *a, int numel_a, int *b, int numel_b)
{
    int *sorted = (int *) malloc((numel_a + numel_b) * sizeof *a);
    int i, a_i = 0, b_i = 0;
    /* merge a[] and b[] into sorted[] */
    for (i = 0; i < (numel_a + numel_b); i++) {
        if (a_i < numel_a && b_i < numel_b) {
            if (a[a_i] < b[b_i]) {
                sorted[i] = a[a_i];
                a_i++;
            } else {
                sorted[i] = b[b_i];
                b_i++;
            }
        } else {
            if (a_i < numel_a) {
                sorted[i] = a[a_i];
                a_i++;
            } else {
                sorted[i] = b[b_i];
                b_i++;
            }
        }
    }
    /* copy sorted[] into a[] */
    memcpy(a, sorted, (numel_a + numel_b) * sizeof *sorted);
    free(sorted);
}

int main(int argc, char **argv)
{
    int rank, size, data[NUM_ELEMENTS];
    MPI_Init(&argc, &argv);
    double t1, t2;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int div = NUM_ELEMENTS / size;
    int numeric = NUM_ELEMENTS % size;
    int rank_div;

    /* Check if last processor */
    if(rank < numeric)
        rank_div = div + 1;
    else
        rank_div = div;
        
    if(rank == 0){
        /* prepare data and display it */
        int i;
        printf("Unsorted:\t");
        for (i = 0; i < NUM_ELEMENTS; i++) {
            data[i] = rand() % NUM_ELEMENTS;
            printf("%d ", data[i]);
        }
        printf("\n");
        
        t1 = MPI_Wtime();   /* take time */
        /* Sort first Half */
        qsort(data, rank_div, sizeof(int), compare_int);

        /* For More than 1 processor */
        int runningTotal = rank_div;
        for(i = 1; i < size; i++){
            if(i < numeric)
                rank_div = div + 1;
            else
                rank_div = div;
            /* Send block to sort */
            MPI_Send(&data[runningTotal], rank_div, MPI_INT, i, i, MPI_COMM_WORLD);
            /* Recieve sorted block */
            MPI_Recv(&data[runningTotal], rank_div, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* Merge sorted blocks */
            merge(data, runningTotal, &data[runningTotal], rank_div);
            runningTotal += rank_div;
        }
        t2 = MPI_Wtime();   /* take time */
        printf("Execution TIme: %f\n", t2-t1);

        /* display sorted array */
        printf("Sorted:\t\t");
        for (i = 0; i < NUM_ELEMENTS; i++)
            printf("%d ", data[i]);
        printf("\n");
    }else{
        /* receive half of the data */
        MPI_Recv(data, rank_div, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* sort the received data */
        qsort(data, rank_div, sizeof(int), compare_int);

        /* send back the sorted data */
        MPI_Send(data, rank_div, MPI_INT, 0, rank, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
   

}
