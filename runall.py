import os
import multiprocessing
import datetime
import sys

usage = "Specify either 'memset' or 'memcpy' as argument"
assert len(sys.argv) > 1, usage
todo = sys.argv[1]

TODOS = {
    "memcpy": (5, set(), 2**32, 500),
    "memset": (4, {0, 32, 2048, 4032}, 2**32, 500)
}
assert todo in TODOS, usage

NCPUS = multiprocessing.cpu_count()
exes = [("{}-erms", 0), ("{}-t", 1), ("{}-nt", 2)]
nthreads = []
for i in range(0, 4):
    cpus = 1 << i
    if cpus > NCPUS:
        continue
    nthreads.append(cpus)

for i in range(1, 30):
    cpus = 16 * i
    if cpus > NCPUS:
        continue
    nthreads.append(cpus)

sizes = [(4096 << x) for x in range(0, 17)]
aligns = [0, 1, 32, 2047, 2048, 2049, 4031, 4032, 4033]
reuses = [x for x in range(0, TODOS[todo][0])]
todo_aligns = TODOS[todo][1]
max_num = TODOS[todo][2]
min_it = TODOS[todo][3]


def os_do(cmd):
    print(cmd)
    os.system(cmd)



date_uid = str(datetime.datetime.now()).replace(" ", "-").replace(":",
                                                                  "-").replace(
                                                                      ".", "-")
DST_FILE = "results-{}.txt".format(date_uid)
if len(sys.argv) > 2:
    DST_FILE = sys.argv[2]
last_align = None
first = "h"
for exe_enum in exes:

    exe = exe_enum[0].format(todo)
    enum = exe_enum[1]
    os_do("gcc -O3 -lpthread -DTODO={} -march=native {}-bench-multi.c -o {}".format(
        enum, todo, exe))
    for nthread in nthreads:
        for size in sizes:
            for reuse in reuses:
                for align in aligns:
                    if len(todo_aligns) != 0:
                        if align not in todo_aligns:
                            continue
                    it = max(int(max_num / size), min_it)
                    os_do("./{} {} {} {} {} {} {} >> {}".format(
                        exe, nthread, size, it, align, reuse, first, DST_FILE))
                    first = "N"
                    if len(sys.argv) > 3:
                        sys.exit(0)
