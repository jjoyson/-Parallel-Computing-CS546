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

#define NUM_ELEMENTS 100

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

    // Create the scount and displs arrys for Scatterv and Gatherv
    int i, *displs, *scount;

    // Find all divisions
    int div = floor((double) NUM_ELEMENTS / (size-1));
    int last_div = NUM_ELEMENTS - div*(size-2);

    // Dynamic the number of divisions
    int recvcount = (rank == size-1) ? last_div : div;
 
    if(rank == 0){
        /* prepare data and display it */
        int i;
        printf("Unsorted:\t");
        for (i = 0; i < NUM_ELEMENTS; i++) {
            data[i] = rand() % NUM_ELEMENTS;
            printf("%d ", data[i]);
        }
        printf("\n");
    }
    t1 = MPI_Wtime();   /* take time */
    // Checking if portioning is needed
    if(size == 1){
        printf("Only 1 processor! Sorting in place!\n");
        qsort(data, NUM_ELEMENTS, sizeof(int), compare_int);
    }
    else{
        // Populate arrys and force proc 0 to not sort
        scount = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        scount[0] = 0;
        displs[0] = 0;
        for(i = 2; i < size; i++){
            scount[i-1] = div;
            displs[i-1] = div*(i-2);
        }
        scount[size-1] = last_div;
        displs[size-1] = div*(size-2);
     
        // Create return buffer
        int *new_data = malloc(sizeof(int) * NUM_ELEMENTS);

        // Scatter arry into portions
        MPI_Scatterv(data, scount, displs, MPI_INT, new_data, recvcount, MPI_INT, 0, MPI_COMM_WORLD);

        // Only sort if not proc 0
        if(rank != 0)
            qsort(new_data, recvcount, sizeof(int), compare_int);
        
        // Gather portions into arry
        MPI_Gatherv(new_data, recvcount, MPI_INT, data, scount, displs, MPI_INT, 0, MPI_COMM_WORLD);

    }

    // Merge portioned arry
    if(rank == 0){
        // Check if merging is needed
        if(size > 1){
            // Skip first proc and start merging from left
            for(i = 3; i < size; i++)
                merge(data, div*(i-2), &data[div*(i-1)], div);
            merge(data, div*(size-2), &data[div*(size-2)], last_div);
        }
        t2 = MPI_Wtime();   /* take time */
        printf("Execution TIme: %f\n", t2-t1);
        
        /* display sorted array */
        printf("Sorted:\t\t");
        for (i = 0; i < NUM_ELEMENTS; i++)
            printf("%d ", data[i]);
        printf("\n");
    }   
    MPI_Finalize();
    return 0;
}
