#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <windows.h>
#define CNTPARAMS 5

const char* names[] = { "XMIN", "XMAX", "N", "M", "T" };
int read_input_file(int** params, double* Sigma) {
    char set[CNTPARAMS + 1] = { 0 };
    FILE* file;
    if (fopen_s(&file, "input.txt", "r") != 0) {
        return -1;
    }

    char line[64] = { 0 }, name[32] = { 0 }, value[32] = { 0 };
    int number = 0;
    int params_read = 0;
    int total_params_needed = CNTPARAMS + 1;

    while (params_read < total_params_needed && fgets(line, sizeof(line), file)) {

        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        int i = 0;
        for (; line[i] != '\n' && line[i] != ' ' && line[i] != '=' && line[i] != '\0'; i++) {
            name[i] = line[i];
        }
        name[i] = '\0';

        for (; line[i] != '=' && line[i] != '\0'; i++);
        if (line[i] == '\0') continue;

        i++;
        while (line[i] == ' ' || line[i] == '\t') i++;

        int j = 0;
        for (; line[i] != '\n' && line[i] != '\r' && line[i] != '\0'; i++) {
            value[j++] = line[i];
        }
        value[j] = '\0';

        if (strcmp(name, "Sigma") == 0) {
            *Sigma = atof(value);
            set[5] = 1;
            params_read++;
            continue;
        }

        number = atoi(value);
        int flag = 1;
        for (j = 0; j < CNTPARAMS; j++) {
            if (!set[j] && strcmp(name, names[j]) == 0) {
                set[j] = 1;
                *params[j] = number;
                flag = 0;
                params_read++;
                break;
            }
        }
        if (flag) {
            fclose(file);
            return -3;
        }
    }
    fclose(file);

    for (int j = 0; j <= CNTPARAMS; j++) {
        if (!set[j]) return -4;
    }
    return 0;
}

void test_read(int* XMIN, int* XMAX, int* N, int* M, int* T, double* Sigma) {
    printf("XMIN: %d\nXMAX: %d\nN: %d\nM: %d\nT: %d\nSigma: %lf\n",
        *XMIN, *XMAX, *N, *M, *T, *Sigma);
}

int write(int t_passed, int N, double* p, double time_sec) {
    FILE* file;
    if (fopen_s(&file, "output.txt", "w") != 0) {
        return -5;
    }

    fprintf(file, "Steps completed: %d\n", t_passed);


    for (int i = 0; i <= N; i++) {
        fprintf(file, "%lf ", p[i]);
    }
    fprintf(file, "\n");
    fclose(file);
    if (fopen_s(&file, "time.txt", "a") != 0) {
        return -6;
    }
    fprintf(file, "N = %d, M = %d\n", N, t_passed);
    fprintf(file, "Time: %.6f seconds\n", time_sec);
    fclose(file);
    return 0;
}

void calculate_de(double* u, double* du, double* k, double* tmp, double* ka, 
                  double sigma, double dx, double dt, int N) {
    int x;
    double coeff = sigma / (dx * dx);
    // k1 = F(u)
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        k[x] = coeff * (u[x+1] - 2.0*u[x] + u[x-1]);
    }
    
    // Сохраняем k1 в ka
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        ka[x] = k[x];  // ka = k1
    }
    
    // k2 = F(u + dt/2 * k1)
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        tmp[x] = u[x] + 0.5 * dt * k[x];
    }
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        k[x] = coeff * (tmp[x+1] - 2.0*tmp[x] + tmp[x-1]);
        ka[x] += 2.0 * k[x];  // ka = k1 + 2*k2
    }
    
    // k3 = F(u + dt/2 * k2)
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        tmp[x] = u[x] + 0.5 * dt * k[x];
    }
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        k[x] = coeff * (tmp[x+1] - 2.0*tmp[x] + tmp[x-1]);
        ka[x] += 2.0 * k[x];  // ka = k1 + 2*k2 + 2*k3
    }
    
    // k4 = F(u + dt * k3)
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        tmp[x] = u[x] + dt * k[x];
    }
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        k[x] = coeff * (tmp[x+1] - 2.0*tmp[x] + tmp[x-1]);
        ka[x] += k[x];  // ka = k1 + 2*k2 + 2*k3 + k4
    }
    
    // u_new = u + dt/6 * ka
    #pragma omp parallel for schedule(static)
    for (x = 1; x < N; x++) {
        du[x] = u[x] + (dt / 6.0) * ka[x];
    }
}

int main() {
    int XMIN = 0, XMAX = 0, N = 0, T = 0, M = 0;
    double Sigma = 0.0;
    int* params[] = { &XMIN, &XMAX, &N, &M, &T };
    int flag = read_input_file(params, &Sigma);
    if (flag != 0) {
        printf("Error reading input file: %d\n", flag);
        test_read(&XMIN, &XMAX, &N, &M, &T, &Sigma);
        return 1;
    }
    if (N <= 0 || M <= 0) {
        printf("Invalid N or M\n");
        return 2;
    }

    double dx = (double)(XMAX - XMIN) / N;
    double dt = (double)T / M;
        double r = Sigma * dt / (dx * dx);
    printf("Stability ratio (r): %lf\n", r);
    if (r > 2.78) {
        printf("WARNING: RK4 scheme may be unstable! (r > 2.78)\n");
        return 3;
    }
    double* p1 = (double*)calloc(N + 1, sizeof(double));
    double* p2 = (double*)calloc(N + 1, sizeof(double));
    double* k = (double*)calloc(N+1, sizeof(double));
    double* tmp = (double*)calloc(N+1, sizeof(double));
    double* ka = (double*)calloc(N+1, sizeof(double));
    double* p = p1;
    if (!p1 || !p2 || !k || !tmp || !ka) {
        printf("Memory allocation failed\n");
        return 4;
    }
        for (int i = 1; i < N; i++) {
        p1[i] = sin(M_PI * dx * i) * XMAX;
    }
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    omp_set_num_threads(4);
    double* cur = p1, * next = p2;
    for (int t = 0; t < M; t++) {
        calculate_de(cur, next, k, tmp, ka, Sigma, dx, dt, N);
        double* tmp_ptr = cur; cur = next; next = tmp_ptr;
    }
    p = cur;
    QueryPerformanceCounter(&end);
    double time_in_seconds = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
    write(M, N, p, time_in_seconds);

    free(p1);
    free(p2);
    free(k);
    free(tmp);
    free(ka);

    printf("Calculation finished successfully. Time: %.6f sec\n", time_in_seconds);
    return 0;
}