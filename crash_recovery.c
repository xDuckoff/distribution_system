#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <signal.h>
#define  Max(a, b) ((a)>(b)?(a):(b))

#define  N  1200
#define NP 4
#define BOSS NP

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, it, rc = 0;
double w = 0.5;
double eps;
double A[N][N] = {};
int nexts[NP] = {};
int prevs[NP] = {};
int nrows[NP] = {};
int start_rows[NP] = {};
int cnt_act;
int corr_it;
void relax();
void init();
void verify();
void boss_proc();
void worker_proc();
void send_matrix();
void send_info();
void set_config();
void print_config();
void try_to_quit();
int who_quit();
void change_plans(int);
void get_new_task();
void give_new_tasks(int);
int max_reduction();
void broadcast();
_Bool fail = 0;

int start_row, last_row, nrow, myrank, nproc;

MPI_Request req[4];
MPI_Status status[4];

int broken_proc, broken_iter;
int backup_iter;
int np_first = 0, np_last = NP - 1, next, prev;


int main(int argc, char** argv)
{
    set_config();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    if(nproc != NP + 1)
    {
        printf("This application is meant to be run with %d processes.\n", NP + 1);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    print_config();

    if (myrank == BOSS) {
        boss_proc();
    } else {
        worker_proc();
    }
}


void set_config() {
    srand(time(NULL));
    broken_proc = rand() % NP+99;
    broken_iter = rand() % itmax + 1;
    backup_iter = 10;
}


void print_config() {
    if (myrank == BOSS) {
        printf("Worker %d will quit on %d iter\n", broken_proc, broken_iter);
        printf("The backup will be saved every %d iterations\n", backup_iter);
        fflush(stdout);
    }
}


void try_to_quit() {
    if (myrank == broken_proc && it == broken_iter) {
        printf("I quit, I'm %d\n", myrank);
        fflush(stdout);
        raise(SIGKILL);
    }
}


void send_matrix() {
    MPI_Send(A[start_row], nrow * N, MPI_DOUBLE, BOSS, 1216, MPI_COMM_WORLD);
}


void collect_matrix() {
    int cur = np_first;
    for (int i = 0; i < cnt_act; ++i) {
        start_row = start_rows[cur];
        nrow = nrows[cur];
        MPI_Recv(A[start_row], N * nrow, MPI_DOUBLE, cur, 1216, MPI_COMM_WORLD, &status[0]);
        cur = nexts[cur];
    }
}


void boss_proc() {
    //init task state
    np_first = 0;
    np_last = NP - 1;

    for (int i = 0; i < NP; ++i) {
        nexts[i] = i + 1;
        prevs[i] = i - 1;
        nrows[i] = N / NP;
        start_rows[i] = (N / NP) * i;
    }
    nexts[NP - 1] = -1;
    prevs[0] = -1;
    corr_it = 0;
    start_row = 0;
    last_row = N;
    init();
    cnt_act = NP;

    //main loop

    it = 1;
    for (; it <= itmax; ++it) {
        /*
         * Проверка, произошла ли ошибка!
         * Если произошла, то восстанавливаемся
         */
        if (!fail) {
            rc = MPI_Barrier(MPI_COMM_WORLD);
            if (rc!=0) {
                int unemployed = who_quit();
                printf("%d quit :(\n", unemployed);
                fflush(stdout);
                change_plans(unemployed);
                give_new_tasks(corr_it);
                it = corr_it;
                fail = 1;
                continue;
            }
        }
        /*
         * Проверка закончена
         * Корректное состояние вычислений достигнуто
         */

        eps = max_reduction();
        broadcast(eps);

        printf("it=%4i   eps=%f\n", it, eps);
        fflush(stdout);

        //Создание backup'a
        if (it % backup_iter == 0) {
            corr_it = it;
            collect_matrix();
        }

        if (eps < maxeps) {
            break;
        }
    }

    collect_matrix();
    verify();

    MPI_Finalize();
}


void worker_proc() {
    start_row = (N / NP) * myrank;
    last_row = (N / NP) * (myrank + 1);
    nrow = last_row - start_row;
    next = myrank + 1;
    prev = myrank - 1;
    init();

    it = 1;
    for (; it <= itmax; it++) {
        try_to_quit();

        eps = 0.;
        relax();

        if (eps < maxeps)
            break;
    }

    send_matrix();
    MPI_Finalize();
}


void relax()
{
    double local_eps = 0.0;
    int dl = 0, dr = 0;
    int unemployed = -1;

    if (myrank == np_first) {
        dl = 1;
    }

    if (myrank == np_last) {
        dr = 1;
    }

    if (myrank != np_first) {
        MPI_Isend(A[start_row], N, MPI_DOUBLE, prev, 1216, MPI_COMM_WORLD, &req[1]);
    }

    if (myrank != np_last) {
        MPI_Isend(A[last_row - 1], N, MPI_DOUBLE, next, 1215, MPI_COMM_WORLD, &req[2]);
    }

    if (myrank != np_last) {
        rc = MPI_Recv(A[last_row], N, MPI_DOUBLE, next, 1216, MPI_COMM_WORLD, &status[3]);
        unemployed = rc != 0 ? next : unemployed;
    }

    if (myrank != np_first) {
        rc = MPI_Recv(A[start_row - 1], N, MPI_DOUBLE, prev, 1215, MPI_COMM_WORLD, &status[0]);
        unemployed = rc != 0 ? prev : unemployed;
    }

    for (i = start_row + dl; i < last_row - dr; ++i) {
        for (j = 1 + i % 2; j <= N - 2; j += 2) {
            double b;
            b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
            A[i][j] = A[i][j] + b;
            local_eps = Max(fabs(b), local_eps);
        }
    }


    if (myrank != np_last) {
        MPI_Isend(A[last_row - 1], N, MPI_DOUBLE, next, 1215, MPI_COMM_WORLD, &req[2]);
    }

    if (myrank != np_first) {
        MPI_Isend(A[start_row], N, MPI_DOUBLE, prev, 1216, MPI_COMM_WORLD, &req[1]);
    }

    if (myrank != np_last) {
        rc = MPI_Recv(A[last_row], N, MPI_DOUBLE, next, 1216, MPI_COMM_WORLD, &status[3]);
        unemployed = rc != 0 ? next : unemployed;
    }

    if (myrank != np_first) {
        rc = MPI_Recv(A[start_row - 1], N, MPI_DOUBLE, prev, 1215, MPI_COMM_WORLD, &status[0]);
        unemployed = rc != 0 ? next : unemployed;
    }


    for (i = start_row + dl; i < last_row - dr; ++i) {
        for (j = 1 + (i + 1) % 2; j <= N - 2; j += 2) {
            double b;
            b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.-A[i][j]);
            A[i][j] = A[i][j]+b;
        }
    }

    if (!fail) {
        rc = MPI_Barrier(MPI_COMM_WORLD);
        if (rc!=0) {
            send_info(unemployed);
            get_new_task();
            eps = 2 * maxeps;
            fail = 1;
            return;
        }
    }

    MPI_Isend(&local_eps, 1, MPI_DOUBLE, BOSS, 8000, MPI_COMM_WORLD, &req[0]);
    MPI_Recv(&eps, 1, MPI_DOUBLE, BOSS, 7000, MPI_COMM_WORLD, &status[0]);

    if (it % backup_iter == 0) {
        send_matrix();
    }
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


void send_info(int u) {
    if (u >= 0) {
        MPI_Isend(&u, 1, MPI_INT, BOSS, 1200, MPI_COMM_WORLD, &req[0]);
    }
}


int who_quit() {
    int res[cnt_act];
    int index;
    int cur = np_first;
    MPI_Request req_test[cnt_act];
    MPI_Status status_test;

    for (int i = 0; i < cnt_act; ++i) {
        MPI_Irecv(&res[i], 1, MPI_INT, cur, 1200, MPI_COMM_WORLD, &req_test[i]);
        cur = nexts[cur];
    }

    index = -1;
    MPI_Waitany(cnt_act, req_test, &index, &status_test);
    return index;
}


void change_plans(int unemployed) {
    int cur = np_first;
    for (int i = 0; i < cnt_act; ++i) {
        if (nexts[cur] == unemployed) {
            nexts[cur] = nexts[unemployed];
        }

        if (prevs[cur] == unemployed) {
            prevs[cur] = prevs[unemployed];
        }
        cur = nexts[cur];
    }

    if (unemployed == np_first) {
        np_first = nexts[np_first];
    }

    if (unemployed == np_last) {
        np_last = prevs[np_last];
    }

    cnt_act--;
    cur = np_first;
    for (int i = 0; i < cnt_act; ++i) {
        nrows[cur] = N / cnt_act;
        start_rows[cur] = (N / cnt_act) * i;
        cur = nexts[cur];
    }
}


void get_new_task() {
    rc = 0;
    MPI_Recv(&next, 1, MPI_INT, BOSS, 2000, MPI_COMM_WORLD, &status[0]);
    MPI_Recv(&prev, 1, MPI_INT, BOSS, 2001, MPI_COMM_WORLD, &status[0]);
    MPI_Recv(&nrow, 1, MPI_INT, BOSS, 2002, MPI_COMM_WORLD, &status[0]);
    MPI_Recv(&start_row, 1, MPI_INT, BOSS, 2003, MPI_COMM_WORLD, &status[0]);
    MPI_Recv(&it, 1, MPI_INT, BOSS, 2004, MPI_COMM_WORLD, &status[0]);
    MPI_Recv(A[start_row], N * nrow, MPI_DOUBLE, BOSS, 2005, MPI_COMM_WORLD, &status[0]);
    MPI_Recv(&np_first, 1, MPI_INT, BOSS, 2006, MPI_COMM_WORLD, &status[0]);
    MPI_Recv(&np_last, 1, MPI_INT, BOSS, 2007, MPI_COMM_WORLD, &status[0]);
    last_row = start_row + nrow;

    printf("Rank %d; start_row %d; last row %d; np_last %d;\n", myrank, start_row, last_row, np_last);
    fflush(stdout);
}


void give_new_tasks(int corr_it) {
    rc = 0;
    int cur = np_first;
    for (int i = 0; i < cnt_act; ++i) {
        MPI_Isend(&nexts[cur], 1, MPI_INT, cur, 2000, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&prevs[cur], 1, MPI_INT, cur, 2001, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&nrows[cur], 1, MPI_INT, cur, 2002, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&start_rows[cur], 1, MPI_INT, cur, 2003, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&corr_it, 1, MPI_INT, cur, 2004, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(A[start_rows[cur]], N * nrows[cur], MPI_DOUBLE, cur, 2005, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&np_first, 1, MPI_INT, cur, 2006, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&np_last, 1, MPI_INT, cur, 2007, MPI_COMM_WORLD, &req[0]);
        cur = nexts[cur];
    }
}


int max_reduction() {
    int cur = np_first;
    double local_eps;

    int res = 0.0;
    for (int i = 0; i < cnt_act; ++i) {
        MPI_Recv(&local_eps, 1, MPI_DOUBLE, cur, 8000, MPI_COMM_WORLD, &status[0]);
        cur = nexts[cur];
        res = Max(res, local_eps);
    }
    return res;
}


void broadcast() {
    int cur = np_first;
    for (int i = 0; i < cnt_act; ++i) {
        MPI_Isend(&eps, 1, MPI_DOUBLE, cur, 7000, MPI_COMM_WORLD, &req[0]);
        cur = nexts[cur];
    }
}