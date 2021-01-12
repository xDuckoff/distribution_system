#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <assert.h>


#define N  8
#define K 10
#define LEN  2 * K * 5

void verify();
void init();
void relax();

int myrank, nproc;
MPI_Request req[2];
MPI_Status status[4];
double message[LEN] = {};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if(nproc != N * N)
    {
        printf("This application is meant to be run with %d^2 processes.\n", N);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    init();

    double time = MPI_Wtime();//start

    relax();

    time = MPI_Wtime() - time; //end
    if (!myrank) {
        printf("Time: %4.2f seconds\n", time);
    }

    verify();
    MPI_Finalize();
    return 0;
}


void init()
{
    if (myrank == 0) {
        //generate message
        for (int i = 0; i < LEN; ++i) {
            message[i] = i;
        }
    }
}


void sender()
{
    enum dist{BOTTOM = N, RIGHT = 1};
    int start_pos = 0;

    for (int iter = 0; iter < LEN / (2 * K); ++iter) {
        //do nothing
        MPI_Barrier(MPI_COMM_WORLD);

        //send
        MPI_Rsend(message + start_pos, K, MPI_DOUBLE, BOTTOM, 0, MPI_COMM_WORLD);
        MPI_Rsend(message + start_pos + K, K, MPI_DOUBLE, RIGHT, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        //do nothing
        start_pos += 2 * K;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //the message was sent, wait
    for (int wait_iter = 0; wait_iter < N - 2; ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


void receiver()
{
    //while nothing comes in, wait
    for (int wait_iter = 0; wait_iter < N - 2; ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    enum src{LEFT = N * N - 2, TOP = N * N - 1 - N};
    int start_pos = 0;

    //message transmission has started
    for (int iter = 0; iter < LEN / (2 * K); ++iter) {
        //ready
        MPI_Irecv(message + start_pos, K, MPI_DOUBLE, LEFT, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(message + start_pos + K, K, MPI_DOUBLE, TOP, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Barrier(MPI_COMM_WORLD);

        //do nothing
        MPI_Barrier(MPI_COMM_WORLD);

        //receive
        MPI_Waitall(2, req, MPI_STATUS_IGNORE);
        start_pos += 2 * K;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


void internal1()
{
    int i = myrank / N;
    int j = myrank % N;
    int pos = i + j;

    //while nothing comes in, wait
    for (int wait_iter = 0; wait_iter < pos / 2; ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //message transmission has started
    int way = (i >= j) ? 0 : 1;
    int next, prev;

    if (way == 0) {
        if (i < N - 1) {
            next = (i + 1) * N;
            prev = (i - 1) * N;
        } else {
            next = i * N + j + 1;
            prev = (j == 0) ? (i - 1) * N : i * N + j - 1;
        }
    } else {
        if (j < N - 1) {
            next = j + 1;
            prev = j - 1;
        } else {
            next = (i + 1) * N + j;
            prev = (i == 0) ? j - 1 : (i - 1) * N + j;
        }
    }

    int start_pos = 0;
    for (int iter = 0; iter < LEN / (2 * K); ++iter) {
        //ready
        MPI_Irecv(message + start_pos + K * way, K, MPI_DOUBLE, prev, way, MPI_COMM_WORLD, &req[0]);
        MPI_Barrier(MPI_COMM_WORLD);

        //receive
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);

        //send
        MPI_Rsend(message + start_pos + K * way, K, MPI_DOUBLE, next, way, MPI_COMM_WORLD);
        start_pos += 2 * K;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //the message was sent, wait
    for (int wait_iter = 0; wait_iter < (N - 2) - (pos / 2); ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


void internal2()
{
    int i = myrank / N;
    int j = myrank % N;
    int pos = i + j;

    //wait
    for (int wait_iter = 0; wait_iter < pos / 2 - 1; ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //message transmission has started
    int way = (i >= j) ? 0 : 1;
    int next, prev;

    if (way == 0) {
        if (i < N - 1) {
            next = (i + 1) * N;
            prev = (i - 1) * N;
        } else {
            next = i * N + j + 1;
            prev = (j == 0) ? (i - 1) * N : i * N + j - 1;
        }
    } else {
        if (j < N - 1) {
            next = j + 1;
            prev = j - 1;
        } else {
            next = (i + 1) * N + j;
            prev = (i == 0) ? j - 1 : (i - 1) * N + j;
        }
    }


    int start_pos = 0;

    //ready
    MPI_Irecv(message + start_pos + K * way, K, MPI_DOUBLE, prev, way, MPI_COMM_WORLD, &req[0]);
    MPI_Barrier(MPI_COMM_WORLD);

    //do nothing
    MPI_Barrier(MPI_COMM_WORLD);

    //receive
    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < LEN / (2 * K) - 1; ++iter) {
        //ready
        MPI_Irecv(message + start_pos + 2 * K + K * way, K, MPI_DOUBLE, prev, way, MPI_COMM_WORLD, &req[0]);
        MPI_Barrier(MPI_COMM_WORLD);

        //send
        MPI_Rsend(message + start_pos + K * way, K, MPI_DOUBLE, next, way, MPI_COMM_WORLD);
        start_pos += 2 * K;
        MPI_Barrier(MPI_COMM_WORLD);

        //receive
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //send
    MPI_Rsend(message + start_pos + K * way, K, MPI_DOUBLE, next, way, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //the message was sent, wait
    for (int wait_iter = 0; wait_iter < (N - 2) - (pos / 2); ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


void internal3()
{
    //The proccess does not participate
    for (int wait_iter = 0; wait_iter < N - 2; ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int wait_iter = 0; wait_iter < LEN / (2 * K); ++wait_iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}



void internal()
{
    int i = myrank / N;
    int j = myrank % N;

    if (i != 0 && j != 0 && i != (N - 1) && j != (N - 1)) {
        internal3();
    } else {
        if ((i + j) % 2 == 0) {
            internal2();
        } else {
            internal1();
        }
    }
}


void relax()
{
    enum role_ranks { SENDER = 0, RECEIVER = N * N - 1 };

    switch (myrank) {
        case SENDER:
        {
            sender();
            break;
        }
        case RECEIVER:
        {
            receiver();
            break;
        }
        default:
        {
            internal();
            break;
        }
    }
}



void verify()
{
    if (myrank == nproc - 1) {
        //verify message
        for (int i = 0; i < LEN; ++i) {
            assert(message[i] == i);
        }
    }
}
