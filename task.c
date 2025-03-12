#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define L 100
#define ITMAX 100

/* макрос для вывода только с процесса 0 */
#define m_printf if(myrank==0) printf

int main(int argc, char **argv) {
    int myrank, ranksize;
    int startrow, lastrow, nrow;
    int i, j, it;
    int ll, shift;
    MPI_Request req[4];
    MPI_Status status[4];
    double t1;
    
    /* Проверка аргументов командной строки */
    if(argc < 3) {
        if(myrank==0) {
            fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Определяем, какие строки матрицы (из L строк) обрабатывает этот процесс */
    startrow = (myrank * L) / ranksize;
    lastrow  = (((myrank + 1) * L) / ranksize) - 1;
    nrow = lastrow - startrow + 1;

    m_printf("JAC1 STARTED\n");

    /* Распределённая матрица B (начальное условие) будет считываться из входного файла.
       На процессе 0 читаем всю матрицу размера L x L. */
    double globalB[L][L]; /* используется только на процессе 0 */
    if(myrank == 0) {
        FILE *fin = fopen(input_filename, "r");
        if(fin == NULL) {
            perror("fopen input");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for(i = 0; i < L; i++) {
            for(j = 0; j < L; j++) {
                if(fscanf(fin, "%lf", &globalB[i][j]) != 1) {
                    fprintf(stderr, "Ошибка чтения элемента (%d,%d)\n", i, j);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        fclose(fin);
    }

    /* Выделяем память для локальной части матрицы B.
       Каждый процесс хранит nrow строк по L элементов */
    double (*B)[L] = malloc(nrow * L * sizeof(double));
    
    /* Подготовка массивов для MPI_Scatterv: на процессе 0 вычисляем,
       сколько строк (и элементов) получает каждый процесс, и смещения */
    int *sendcounts = NULL, *displs = NULL;
    if(myrank == 0) {
        sendcounts = malloc(ranksize * sizeof(int));
        displs = malloc(ranksize * sizeof(int));
        for (int p = 0; p < ranksize; p++) {
            int start = (p * L) / ranksize;
            int end = (((p + 1) * L) / ranksize) - 1;
            int count = end - start + 1;
            sendcounts[p] = count * L;
            if(p == 0)
                displs[p] = 0;
            else
                displs[p] = displs[p - 1] + sendcounts[p - 1];
        }
    }
    /* Распределяем строки матрицы B: каждый процесс получает свою часть */
    MPI_Scatterv(myrank == 0 ? globalB : NULL, sendcounts, displs, MPI_DOUBLE,
                 B, nrow * L, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    /* Выделяем память для матрицы A с запасом для "граничных" строк:
       nrow строк + 2 дополнительных для обмена */
    double (*A)[L] = malloc((nrow + 2) * L * sizeof(double));
    /* Инициализируем строки A[1..nrow] нулями */
    for(i = 1; i <= nrow; i++) {
        for(j = 0; j < L; j++) {
            A[i][j] = 0.0;
        }
    }
    
    /* Итерационный цикл: выполняется ITMAX итераций.
       Обновление значений локальной матрицы B по схеме Жакоби */
    t1 = MPI_Wtime();
    for(it = 1; it <= ITMAX; it++) {
        /* Копируем внутренние элементы B в A (кроме граничных, которые получают через обмен) */
        for(i = 1; i <= nrow; i++) {
            if (((i == 1) && (myrank == 0)) || ((i == nrow) && (myrank == ranksize - 1)))
                continue;
            for(j = 1; j <= L - 2; j++) {
                A[i][j] = B[i - 1][j];
            }
        }
        
        /* Обмен граничными строками между процессами */
        if(myrank != 0)
            MPI_Irecv(&A[0][0], L, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req[0]);
        if(myrank != ranksize - 1)
            MPI_Isend(&A[nrow][0], L, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req[2]);
        if(myrank != ranksize - 1)
            MPI_Irecv(&A[nrow + 1][0], L, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req[3]);
        if(myrank != 0)
            MPI_Isend(&A[1][0], L, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req[1]);
        ll = 4; shift = 0;
        if(myrank == 0) { ll = 2; shift = 2; }
        if(myrank == ranksize - 1) { ll = 2; }
        MPI_Waitall(ll, &req[shift], status);
        
        /* Обновляем локальную матрицу B по схеме Жакоби */
        for(i = 1; i <= nrow; i++) {
            if (((i == 1) && (myrank == 0)) || ((i == nrow) && (myrank == ranksize - 1)))
                continue;
            for(j = 1; j <= L - 2; j++)
                B[i - 1][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.0;
        }
    }
    double local_time = MPI_Wtime() - t1;
    printf("%d: Time of task = %lf\n", myrank, local_time);

    /* Собираем обновлённую матрицу B с всех процессов в глобальный массив на процессе 0 */
    double *globalResult = NULL;
    if(myrank == 0) {
        globalResult = malloc(L * L * sizeof(double));
    }
    MPI_Gatherv(B, nrow * L, MPI_DOUBLE,
                globalResult, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Процесс 0 записывает результат в выходной файл */
    if(myrank == 0) {
        FILE *fout = fopen(output_filename, "w");
        if(fout == NULL) {
            perror("fopen output");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for(i = 0; i < L; i++) {
            for(j = 0; j < L; j++) {
                fprintf(fout, "%lf ", globalResult[i * L + j]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
        free(globalResult);
        free(sendcounts);
        free(displs);
    }
    
    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}
