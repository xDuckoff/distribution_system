#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#define  Max(a, b) ((a)>(b)?(a):(b))

#define  N  1200

double maxeps = 0.1e-7;
int itmax = 100;
int i, j;
double w = 0.5;
double eps;
double A[N][N] = {};
void relax();
void init();
void verify();

int start_row, last_row, nrow, myrank, nproc;
MPI_Request req[4];
MPI_Status status[4];

int main(int argc, char** argv)
{
    int it;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    start_row = (N / nproc) * myrank;
    last_row = (N / nproc) * (myrank + 1);

    nrow = last_row - start_row;
    init();

    double time = MPI_Wtime();//start
    for (it = 1; it <= itmax; it++) {
        eps = 0.;
        relax();
        if (!myrank) {
            printf("it=%4i   eps=%f\n", it, eps);
        }

        if (eps < maxeps)
            break;
    }

    time = MPI_Wtime() - time; //end


    MPI_Barrier(MPI_COMM_WORLD);
    if (!myrank) {
        MPI_Gather(MPI_IN_PLACE, nrow * N,
                MPI_DOUBLE, A, nrow * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(A[start_row], nrow * N,
                MPI_DOUBLE, A, nrow * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (!myrank) {
        verify();
        printf("%d\n Time: %4.2f seconds\n", N, time);
    }

    MPI_Finalize();
    return 0;
}

void init()
{
    for (i = start_row; i < last_row; ++i) {
        if (i == 0 || i == N - 1) {
            continue;
        }

        for (j = 1; j < N - 1; ++j) {
            A[i][j] = (1. + i + j);
        }
    }
}

void relax()
{
    double local_eps = 0.0;
    int dl = 0, dr = 0;

    if (!myrank) {
        dl = 1;
    }

    if (myrank == nproc - 1) {
        dr = 1;
    }

    if (myrank) {
        MPI_Irecv(A[start_row - 1], N, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req[0]);
    }

    if (myrank != nproc - 1) {
        MPI_Isend(A[last_row - 1], N, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req[2]);
    }

    if (myrank != nproc - 1) {
        MPI_Irecv(A[last_row], N, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req[3]);
    }

    if (myrank) {
        MPI_Isend(A[start_row], N, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req[1]);
    }

    MPI_Waitall(4 - (dl + dr) * 2, &req[dl * 2], status);

    for (i = start_row + dl; i < last_row - dr; ++i) {
        for (j = 1 + i % 2; j <= N - 2; j += 2) {
            double b;
            b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
            A[i][j] = A[i][j] + b;
            local_eps = Max(fabs(b), local_eps);
        }
    }

    if (myrank != 0) {
        MPI_Irecv(A[start_row - 1], N, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req[0]);
    }

    if (myrank != nproc - 1) {
        MPI_Isend(A[last_row - 1], N, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req[2]);
    }

    if (myrank != nproc - 1) {
        MPI_Irecv(A[last_row], N, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req[3]);
    }

    if (myrank != 0) {
        MPI_Isend(A[start_row], N, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req[1]);
    }

    MPI_Waitall(4 - (dl + dr) * 2, &req[dl * 2], status);


    for (i = start_row + dl; i < last_row - dr; ++i) {
        for (j = 1 + (i + 1) % 2; j <= N - 2; j += 2) {
            double b;
            b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.-A[i][j]);
            A[i][j] = A[i][j]+b;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void verify()
{
    double s;

    s = 0.;

    for (i = 0; i<=N - 1; ++i) {
        for (j = 0; j<= N - 1; ++j) {
            s += A[i][j] * (i + 1) * (j + 1) / (N * N);
        }
    }

    printf("  S = %f\n", s);
}
