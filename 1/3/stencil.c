/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using a nonblocking send/receive with manual packing/unpacking.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations to exchange a halo with
 * neighbors. Grid points in a halo are packed and unpacked before and after communications.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

int ind_f(int i, int j, int bx)
{
    return ind(i, j);
}

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag);

void init_sources(int bx, int by, int offx, int offy, int n,
                  const int nsources, int sources[][2], int *locnsources_ptr, int locsources[][2]);

void alloc_bufs(int bx, int by,
                double **atemp_ptr, double **atempr_ptr,
                double **aold_ptr, double **anew_ptr);

void reupdate_data(int bx, int by, double *aold,
                 double *atemp);

void update_data(int bx, int by, double *aold,
                 double *atempr);

void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr);

void free_bufs(double *aold, double *anew, double *atemp, double *atempr);

int main(int argc, char **argv)
{
    int rank, size;
    int n, energy, niters, px, py;

    int rx, ry;
    int north, south, west, east;
    int bx, by, offx, offy;

    /* three heat sources */
    const int nsources = 3;
    int sources[nsources][2];
    int locnsources;            /* number of sources in my area */
    int locsources[nsources][2];        /* sources local to my rank */

    double t1, t2;

    int iter, i;

    double *aold, *anew, *tmp, *atemp, *atempr;

    double heat, rheat;

    int final_flag;

    /* initialize MPI envrionment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument checking and setting */
    setup(rank, size, argc, argv, &n, &energy, &niters, &px, &py, &final_flag);

    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

    /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
    rx = rank % px;
    ry = rank / px;

    /* determine my four neighbors */
    north = (ry - 1) * px + rx;
    if (ry - 1 < 0)
        north = MPI_PROC_NULL;
    south = (ry + 1) * px + rx;
    if (ry + 1 >= py)
        south = MPI_PROC_NULL;
    west = ry * px + rx - 1;
    if (rx - 1 < 0)
        west = MPI_PROC_NULL;
    east = ry * px + rx + 1;
    if (rx + 1 >= px)
        east = MPI_PROC_NULL;

    /* decompose the domain */
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */
    offx = rx * bx;     /* offset in x */
    offy = ry * by;     /* offset in y */

    /* printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank, ry,rx,west,east,north,south); */

    /* initialize three heat sources */
    init_sources(bx, by, offx, offy, n, nsources, sources, &locnsources, locsources);

    /* allocate working arrays & communication buffers */
    alloc_bufs(bx, by, &atemp, &atempr, &aold, &anew);

    t1 = MPI_Wtime();   /* take time */

    for (iter = 0; iter < niters; ++iter) {

        /* refresh heat sources */
        for (i = 0; i < locnsources; ++i) {
            aold[ind(locsources[i][0], locsources[i][1])] += energy;    /* heat source */
        }

        /* Instantiate and Initialize scountsc n dis array */
        int scountsx [size];
        int dis [size];

        for(i = 0; i < size; i++){
            scountsx[i] = 0;
            dis[i] = 0;
        }

        /* Update data & dis per proc */
        if(north != MPI_PROC_NULL){
            scountsx[north] = bx;
        }
        if(south != MPI_PROC_NULL){
            scountsx[south] = bx;
            dis[south] = bx;
        }
        if(east != MPI_PROC_NULL){
            scountsx[east] = by;
            dis[east] = 2*bx;
        }
        if(west != MPI_PROC_NULL){
            scountsx[west] = by;
            dis[west] = 2*bx + by;
        }

        /* Pack arry with data */
        reupdate_data(bx, by, aold, atemp);

        MPI_Alltoallv(atemp, scountsx, dis, MPI_DOUBLE, atempr, scountsx, dis, MPI_DOUBLE, MPI_COMM_WORLD);

        /* update data */
        update_data(bx, by, aold, atempr);

        /* update grid points */
        update_grid(bx, by, aold, anew, &heat);

        /* swap working arrays */
        tmp = anew;
        anew = aold;
        aold = tmp;
    }

    t2 = MPI_Wtime();

    /* free working arrays and communication buffers */
    free_bufs(aold, anew, atemp, atempr);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (!rank)
        printf("[%i] last heat: %f time: %f\n", rank, rheat, t2 - t1);

    MPI_Finalize();
    return 0;
}

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag)
{
    int n, energy, niters, px, py;

    (*final_flag) = 0;

    if (argc < 6) {
        if (!rank)
            printf("usage: stencil_mpi <n> <energy> <niters> <px> <py>\n");
        (*final_flag) = 1;
        return;
    }

    n = atoi(argv[1]);  /* nxn grid */
    energy = atoi(argv[2]);     /* energy to be injected per iteration */
    niters = atoi(argv[3]);     /* number of iterations */
    px = atoi(argv[4]); /* 1st dim processes */
    py = atoi(argv[5]); /* 2nd dim processes */

    if (px * py != proc)
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort if px or py are wrong */
    if (n % px != 0)
        MPI_Abort(MPI_COMM_WORLD, 2);   /* abort px needs to divide n */
    if (n % py != 0)
        MPI_Abort(MPI_COMM_WORLD, 3);   /* abort py needs to divide n */

    (*n_ptr) = n;
    (*energy_ptr) = energy;
    (*niters_ptr) = niters;
    (*px_ptr) = px;
    (*py_ptr) = py;
}

void init_sources(int bx, int by, int offx, int offy, int n,
                  const int nsources, int sources[][2], int *locnsources_ptr, int locsources[][2])
{
    int i, locnsources = 0;

    sources[0][0] = n / 2;
    sources[0][1] = n / 2;
    sources[1][0] = n / 3;
    sources[1][1] = n / 3;
    sources[2][0] = n * 4 / 5;
    sources[2][1] = n * 8 / 9;

    for (i = 0; i < nsources; ++i) {    /* determine which sources are in my patch */
        int locx = sources[i][0] - offx;
        int locy = sources[i][1] - offy;
        if (locx >= 0 && locx < bx && locy >= 0 && locy < by) {
            locsources[locnsources][0] = locx + 1;      /* offset by halo zone */
            locsources[locnsources][1] = locy + 1;      /* offset by halo zone */
            locnsources++;
        }
    }

    (*locnsources_ptr) = locnsources;
}

void alloc_bufs(int bx, int by, double **atemp_ptr, double **atempr_ptr, double **aold_ptr, double **anew_ptr)
{
    double *aold, *anew, *atemp, *atempr;

    /* allocate two working arrays */
    anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */

    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    /* allocate communication buffers */
    atemp = (double *) malloc((bx+by) * 2 * sizeof(double));
    atempr = (double *) malloc((bx*by) * 2 * sizeof(double));

    memset(atemp, 0, (by + bx) * 2 * sizeof(double));
    memset(atempr, 0, (by + bx) * 2 * sizeof(double));

    (*aold_ptr) = aold;
    (*anew_ptr) = anew;
    (*atemp_ptr) = atemp;
    (*atempr_ptr) = atempr;
}

void free_bufs(double *aold, double *anew, double *atemp, double *atempr)
{
    free(aold);
    free(anew);
    free(atemp);
    free(atempr);
}

void reupdate_data(int bx, int by, double *aold,
                 double *atemp)
{
    int i;
    int j = 0;
    for (i = 0; i < bx; ++i){
        atemp[j] = aold[ind(i + 1, 1)];     /* #1 row */
        j++;
    }
    for (i = 0; i < bx; ++i){
        atemp[j] = aold[ind(i + 1, by)];    /* #(by) row */
        j++;
    }
    for (i = 0; i < by; ++i){
        atemp[j] = aold[ind(bx, i + 1)];     /* #(bx) col */
        j++;
    }
    for (i = 0; i < by; ++i){
        atemp[j] = aold[ind(1, i + 1)];      /* #1 col */
        j++;
    }
}

void update_data(int bx, int by, double *aold,
                 double *atempr)
{
    int i;
    int j = 0;
    for (i = 0; i < bx; ++i){
        aold[ind(i + 1, 0)] = atempr[j];     /* #0 row */
        j++;
    }
    for (i = 0; i < bx; ++i){
        aold[ind(i + 1, by + 1)] = atempr[j];        /* #(by+1) row */
        j++;
    }
    for (i = 0; i < by; ++i){
        aold[ind(bx + 1, i + 1)] = atempr[j]; /* #(bx+1) col */
        j++;
    }
    for (i = 0; i < by; ++i){
        aold[ind(0, i + 1)] = atempr[j];      /* #0 col */
        j++;
    }
}

void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr)
{
    int i, j;
    double heat = 0.0;

    for (i = 1; i < bx + 1; ++i) {
        for (j = 1; j < by + 1; ++j) {
            anew[ind(i, j)] =
                anew[ind(i, j)] / 2.0 + (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                         aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;
            heat += anew[ind(i, j)];
        }
    }

    (*heat_ptr) = heat;
}
