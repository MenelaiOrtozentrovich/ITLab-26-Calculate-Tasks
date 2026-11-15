#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define CNTPARAMS 5

#ifdef _MSC_VER
//#include <intrin.h> // RDTSC
#include <windows.h> // Для QueryPerformanceCounter
#endif
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

#ifdef _MSC_VER
void bind_to_pcores_no_ht() {
    DWORD bufferSize = 0;
    GetLogicalProcessorInformation(NULL, &bufferSize);
    
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = 
        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(bufferSize);
    GetLogicalProcessorInformation(buffer, &bufferSize);
    
    DWORD_PTR pcoreMask = 0;
    int pcoreCount = 0;
    
    for (DWORD i = 0; i < bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); i++) {
        if (buffer[i].Relationship == RelationProcessorCore) {
            DWORD_PTR coreMask = buffer[i].ProcessorMask;
            DWORD_PTR lowestBit = coreMask & (~coreMask + 1);
            
            if (pcoreCount < 4) {
                pcoreMask |= lowestBit;
                pcoreCount++;
            }
        }
    }
    
    free(buffer);
    SetProcessAffinityMask(GetCurrentProcess(), pcoreMask);
    printf("Bound to %d P-cores (no HT), mask = 0x%I64X\n", 
           pcoreCount, (unsigned long long)pcoreMask);
}
#endif

int main() {
    #ifdef _MSC_VER
    bind_to_pcores_no_ht();
#endif
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

    printf("Sigma: %lf, dx: %lf, dt: %lf\n", Sigma, dx, dt);
    printf("Stability ratio (r): %lf\n", r);
    if (r > 0.5) {
        printf("WARNING: Scheme is unstable! (r > 0.5)\n");
    }

    double* p1 = (double*)calloc(N + 1, sizeof(double));
    double* p2 = (double*)calloc(N + 1, sizeof(double));
    double* p = p1;

    if (!p1 || !p2) {
        printf("Memory allocation failed\n");
        return 3;
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
        int x;
        #pragma omp parallel for schedule(static)
        for (x = 1; x < N; x++){
            
            next[x] = cur[x] + r * (cur[x + 1] - 2.0 * cur[x] + cur[x - 1]);
        }
        double* tmp = cur; cur = next; next = tmp;
    }
    p = cur;

    QueryPerformanceCounter(&end);
    double time_in_seconds = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;

    write(M, N, p, time_in_seconds);

    free(p1);
    free(p2);

    printf("Calculation finished successfully. Time: %.6f sec\n", time_in_seconds);
    return 0;
}
