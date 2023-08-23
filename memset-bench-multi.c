#include <assert.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>
// gcc -DI_NO_HIDDEN_LABELS -O3 -march=native memset-bench-multi.c -o
// memset-bench-multi
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define NEVER_INLINE  __attribute__((noinline))
#define CONST_ATTR    __attribute__((const))
#define PURE_ATTR     __attribute__((pure))
#define BENCH_FUNC    __attribute__((noinline, noclone, aligned(4096)))

#define COMPILER_OOE_BARRIER() asm volatile("lfence" : : : "memory")
#define OOE_BARRIER()          asm volatile("lfence" : : :)
#define COMPILER_BARRIER()     asm volatile("" : : : "memory");
#define COMPILER_DO_NOT_OPTIMIZE_OUT(X)                                        \
 asm volatile("" : : "i,r,m,v"(X) : "memory")

#define _CAT(X, Y)   X##Y
#define CAT(X, Y)    _CAT(X, Y)
#define _V_TO_STR(X) #X
#define V_TO_STR(X)  _V_TO_STR(X)

#define NO_LSD_RD(tmp, r) "pop " #tmp "\nmovq " #r ", %%rsp\n"
#define NO_LSD_WR(tmp, r) "push " #tmp "\nmovq " #r ", %%rsp\n"

#define IMPOSSIBLE(X)                                                          \
 if (X) {                                                                      \
  __builtin_unreachable();                                                     \
 }

#define PRINT(...) fprintf(stderr, __VA_ARGS__)

static struct timespec
get_ts() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

static uint64_t
to_ns(struct timespec ts) {
    return ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
}
static uint64_t
get_ns() {
    struct timespec ts = get_ts();
    return to_ns(ts);
}


static uint64_t
get_ns_dif(struct timespec ts_start, struct timespec ts_end) {
    return to_ns(ts_end) - to_ns(ts_start);
}

static double
gb_sec(uint64_t bytes, uint64_t ns) {
    double d_bytes = bytes;
    double d_ns    = ns;
    double d_gb    = d_bytes / ((double)1024 * 1024 * 1024);
    double d_sec   = d_ns / (double)(1000 * 1000 * 1000);
    return d_gb / d_sec;
}


static uint32_t          g_iter;
static uint32_t          g_reuse;
static size_t            g_size;
static pthread_barrier_t g_barrier;

typedef struct targs {
    uint8_t * dst;
    uint8_t   val;

    pthread_t tid;
    uint64_t  ns_out;

    uint8_t * sink;
} targs_t;

#define ssse3     __memset_ssse3_back
#define erms      __memset_erms
#define sse2_erms __memset_sse2_unaligned_erms
#define sse2      __memset_sse2_unaligned
#define evex_erms __memset_evex_unaligned_erms
#define evex      __memset_evex_unaligned


static void
memset_t(uint8_t * dst, uint8_t val, size_t len) {
    __m256i v0 = _mm256_set1_epi8(val);
#define VEC_SIZE "32"
    // clang-format off
    __asm__ volatile(
        "1:\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 0)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 1)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 2)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 3)(%[dst])\n"
        "subq $(" VEC_SIZE " * -4), %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        "jg 1b"
        :  [dst] "+r"(dst), [len] "+r"(len)
        :  [v0] "v"(v0)
        : "cc", "memory");
    // clang-format on
#undef VEC_SIZE
};

static void
memset_nt(uint8_t * dst, uint8_t val, size_t len) {
    __m256i v0 = _mm256_set1_epi8(val);
#define VEC_SIZE "32"
    // clang-format off
    __asm__ volatile(
        "1:\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 0)(%[dst])\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 1)(%[dst])\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 2)(%[dst])\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 3)(%[dst])\n"
        "subq $(" VEC_SIZE " * -4), %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        "jg 1b"
        :  [dst] "+r"(dst), [len] "+r"(len)
        :  [v0] "v"(v0)
        : "cc", "memory");
    // clang-format on
#undef VEC_SIZE
};


static void
memset_erms(uint8_t * dst, uint8_t val, size_t len) {
    __asm__ volatile("rep stosb" : "+D"(dst), "+c"(len) : "a"(val) : "memory");
}


static uint8_t
use(uint8_t * dst, size_t len) {
    __m256i v0, v1, v2, v3;
#define VEC_SIZE "32"
    // clang-format off
    __asm__ volatile(
        "vpxor %[v0], %[v0], %[v0]\n"
        "vpxor %[v1], %[v1], %[v1]\n"
        "vpxor %[v2], %[v2], %[v2]\n"
        "vpxor %[v3], %[v3], %[v3]\n"
        "1:\n"
        "vpxor (" VEC_SIZE " * 0)(%[dst]), %[v0]\n"
        "vpxor (" VEC_SIZE " * 1)(%[dst]), %[v1]\n"
        "vpxor (" VEC_SIZE " * 2)(%[dst]), %[v2]\n"
        "vpxor (" VEC_SIZE " * 3)(%[dst]), %[v3]\n"
        "subq $(" VEC_SIZE " * -4), %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        "jg 1b"
        "vpxor %[v0], %[v1], %[v1]\n"
        "vpxor %[v2], %[v3], %[v3]\n"
        "vpxor %[v1], %[v3], %[v3]\n"
        "vpcmpeqb %[v0], %[v0], %[v0]\n"
        "vpor %[v0], %[v3], %[v3]\n"
        "vmovq %[v3], %[len]\n"
        "incq %[len]\n"
        : [dst] "+r"(dst),  [len] "+r"(len), [v0] "=&v"(v0),
          [v1] "=&v"(v1), [v2] "=&v"(v2), [v3] "=&v"(v3)
        :
        : "cc", "memory");
    // clang-format on
#undef VEC_SIZE
    return len;
};
#ifndef TODO
# define TODO 3
#endif
#if TODO == 0
# define FUNC memset_erms
#elif TODO == 1
# define FUNC memset_t
#elif TODO == 2
# define FUNC memset_nt
#else
# error "Unknown TODO"
#endif

#define NAME V_TO_STR(FUNC)


void * BENCH_FUNC
bench(void * arg) {
    struct timespec start, end;
    uint32_t        bench_iter = g_iter;
    uint32_t        reuse      = g_reuse;
    size_t          sz         = g_size;
    bench_iter &= -1;

    uint8_t * dst  = ((targs_t *)arg)->dst;
    uint8_t   val  = ((targs_t *)arg)->val;
    uint8_t * sink = ((targs_t *)arg)->sink;
    assert(reuse <= 3);

    switch (reuse) {
        case 0:
            pthread_barrier_wait(&g_barrier);
            start = get_ts();
            __asm__ volatile(".p2align 6\n" : : :);
            for (; bench_iter; --bench_iter) {
                FUNC(dst, val, sz);
            }
            end = get_ts();
            break;
        case 1:
            pthread_barrier_wait(&g_barrier);
            start = get_ts();
            __asm__ volatile(".p2align 6\n" : : :);
            for (; bench_iter; bench_iter -= 2) {
                FUNC(dst, val, sz);
                COMPILER_DO_NOT_OPTIMIZE_OUT(memcpy(sink, dst, sz));
            }
            end = get_ts();
            break;
        case 2:
            pthread_barrier_wait(&g_barrier);
            start = get_ts();
            __asm__ volatile(".p2align 6\n" : : :);
            for (; bench_iter; bench_iter -= 2) {
                FUNC(dst, val, sz);
                COMPILER_DO_NOT_OPTIMIZE_OUT(use(dst, sz));
            }
            end = get_ts();
            break;
        case 3:
            pthread_barrier_wait(&g_barrier);
            start = get_ts();
            __asm__ volatile(".p2align 6\n" : : :);
            for (; bench_iter; bench_iter -= 2) {
                FUNC(dst, val, sz);
                val |= use(dst, sz);
            }
            end = get_ts();
            break;
        default:
            __builtin_unreachable();
    }

    ((targs_t *)arg)->ns_out = get_ns_dif(start, end);
    return NULL;
}


int
main(int argc, char ** argv) {
    if (argc < 5) {
        printf("Usage: %s <nthreads> <size> <iter> <opt:align> <opt:reuse>\n",
               argv[0]);
        return 0;
    }
    fprintf(stderr, "Getting Conf\n");

    long   nthreads = strtol(argv[1], NULL, 10);
    char * end;
    size_t size = strtoul(argv[2], &end, 10);

    if (strncasecmp(end, "kb", 2) == 0) {
        size *= 1024;
    }
    else if (strncasecmp(end, "mb", 2) == 0) {
        size *= 1024 * 1024;
    }
    else if (strncasecmp(end, "gb", 2) == 0) {
        size *= 1024 * 1024 * 1024;
    }
    uint32_t iter = strtoul(argv[3], NULL, 10);
    iter &= -1;
    uint32_t align = argc >= 5 ? strtoul(argv[4], NULL, 10) : 0;
    uint32_t reuse = argc >= 6 ? strtoul(argv[5], NULL, 10) : 0;

    fprintf(stderr, "Setting of threads\n");
    assert(reuse <= 3);
    align %= 4096;
    align &= -32;
    assert(nthreads != 0 && size != 0 && iter != 0);
    if (nthreads < 0) {
        nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    }

    g_iter = iter;
    g_size = size;
    assert(pthread_barrier_init(&g_barrier, NULL, nthreads) == 0);

    pthread_attr_t attr;

    assert(pthread_attr_init(&attr) == 0);
    assert(pthread_attr_setstacksize(&attr, 16384) == 0);

    targs_t targs[nthreads];
    fprintf(stderr, "Running\n");
    for (uint8_t val = 0; val < 2; ++val) {
        for (long i = 0; i < nthreads; ++i) {
            fprintf(stderr, "Creating Thread: %lu\n", i);
            uint8_t * dst =
                (uint8_t *)mmap(NULL, size + align, PROT_READ | PROT_WRITE,
                                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0L);
            uint8_t * sink =
                (uint8_t *)mmap(NULL, size, PROT_READ | PROT_WRITE,
                                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0L);
            assert(sink != MAP_FAILED);
            assert(dst != MAP_FAILED);
            assert((align % 32) == 0);
            targs[i].dst  = dst + align;
            targs[i].val  = val;
            targs[i].sink = sink;
            assert(pthread_create(&(targs[i].tid), &attr, bench,
                                  (void *)(targs + i)) == 0);
        }

        for (long i = 0; i < nthreads; ++i) {
            assert(pthread_join(targs[i].tid, NULL) == 0);
        }
        for (long i = 0; i < nthreads; ++i) {
            fprintf(stderr, "Unmapping Thread: %lu\n", i);
            munmap(targs[i].dst, size + align);
            munmap(targs[i].sink, size);
        }
    }

#if 0
    printf("Nthreads = %ld, Size = %zu Bytes, Iter = %u, Align = %u\n", nthreads, size,
           iter, align);
    printf("Impl: %s\n", NAME);
    double sum   = 0.0;
    size_t bytes = size * iter;
    for (long i = 0; i < nthreads; ++i) {
        double bw = gb_sec(bytes, targs[i].ns_out);
        printf("Thread %-2ld: %.4lf GB/sec\n", i, bw);
        sum += bw;
    }
    printf("Average: %.4lf GB/sec\n", sum / ((double)nthreads));
#else
    printf("func,nthreads,iter,size,align,reuse,time_ns\n");
    for (long i = 0; i < nthreads; ++i) {
        printf("%s,%ld,%u,%zu,%u,%u,%lu\n", NAME, nthreads, iter, size, align,
               reuse, targs[i].ns_out);
    }
#endif
}
