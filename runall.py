import os
import multiprocessing
import datetime
import sys

NCPUS = multiprocessing.cpu_count()
exes = [("memcpy-erms", 0), ("memcpy-t", 1), ("memcpy-nt", 2)]
nthreads = []
for i in range(0, 3):
    cpus = 1 << i
    if cpus > NCPUS:
        continue
    nthreads.append(cpus)

for i in range(1, 30):
    cpus = 8 * i
    if cpus > NCPUS:
        continue
    nthreads.append(cpus)

sizes = [(4096 << x) for x in range(0, 1)]
aligns = [0, 1, 32, 2047, 2048, 2049, 4031, 4032, 4033]
reuses = [0, 1, 2, 3, 4, 5]


def os_do(cmd):
    print(cmd)
    os.system(cmd)


date_uid = str(datetime.datetime.now()).replace(" ", "-").replace(":",
                                                                  "-").replace(
                                                                      ".", "-")
DST_FILE = "results-{}.txt".format(date_uid)
if len(sys.argv) > 1:
    DST_FILE = sys.argv[1]

for exe_enum in exes:

    exe = exe_enum[0]
    enum = exe_enum[1]
    os_do("gcc -O3 -DTODO={} -march=native memcpy-bench-multi.c -o {}".format(
        enum, exe))
    for nthread in nthreads:
        for size in sizes:
            for reuse in reuses:
                for align in aligns:
                    it = max(int((2**32) / size), 500)
                    os_do("./{} {} {} {} {} {} >> {}".format(
                        exe, nthread, size, it, align, reuse, DST_FILE))
