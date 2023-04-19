import sys
import statistics


def _arg_to_list(arg):
    if isinstance(arg, list) or isinstance(arg, tuple):
        out = []
        for arg_item in arg:
            out += _arg_to_list(arg_item)
        return out
    else:
        return [arg]


def dict_key(*args):
    return "-|-".join(map(str, _arg_to_list(args)))


def strs(s, substrs=None):
    if substrs is None:
        substrs = []
    elif isinstance(substrs, str):
        substrs = [substrs]
    s = str(s)
    slen = 0
    while slen != len(s):
        slen = len(s)
        s = s.lstrip().rstrip()
        for substr in substrs:
            s = s.lstrip(substr).rstrip(substr)
    return s


def split_line(line):
    line = strs(line).split(",")
    return line[0], int(line[1]), int(line[2]), int(line[3]), int(
        line[4]), int(line[5]), float(line[6])


def line_key(line):
    func, nthread, iter, size, align, ops, unused = split_line(line)
    return item_key(func, nthread, iter, size, align, ops)


def item_key(func, nthread, iter, size, align, ops):
    return dict_key(func, nthread, size, align, ops)


def ns_to_sec(ns):
    return ns / (1000 * 1000 * 1000)


class Result():

    def __init__(self, line):
        func, nthread, iter, size, align, ops, unused = split_line(line)

        self.func = strs(func)
        self.nthread = int(nthread)
        self.iter = int(iter)
        self.size = int(size)
        self.align = int(align)
        self.ops = int(ops)
        self.times = []

        assert self.nthread > 0 and self.nthread <= 96
        assert self.iter >= 500
        assert (self.size & (self.size - 1)) == 0
        assert self.align >= 0 and self.align < 4096

        self.add_line(line)

    def is_done(self):
        return len(self.times) == self.nthread

    def add_line(self, line):
        assert not self.is_done()
        func, nthread, iter, size, align, ops, time = split_line(line)

        assert self.func == strs(func)
        assert self.nthread == int(nthread)
        assert self.iter == int(iter)
        assert self.size == int(size)
        assert self.align == int(align)
        assert self.ops == int(ops)

        self.times.append(float(time))

    def get_gb(self):
        return (self.iter * self.size) / (1024 * 1024 * 1024)

    def get_bw(self):
        assert self.is_done()
        gb = self.get_gb()

        bw_times = []
        for time in self.times:
            bw_times.append(gb / ns_to_sec(time))

        return bw_times

    def get_stat(self):
        return statistics.geometric_mean(self.times)


class Results():

    def __init__(self):
        self.results_ = {}
        self.keys_ = {}

    def add_line(self, line):
        if not line.startswith("memcpy"):
            return

        key = line_key(line)
        if key in self.results_:
            self.results_[key].add_line(line)
        else:
            self.results_[key] = Result(line)

        func, nthread, iter, size, align, ops, unused = split_line(line)

        self.keys_.setdefault("funcs", set()).add(func)
        self.keys_.setdefault("nthreads", set()).add(nthread)
        self.keys_.setdefault("iters", set()).add(iter)
        self.keys_.setdefault("sizes", set()).add(size)
        self.keys_.setdefault("alignments", set()).add(align)
        self.keys_.setdefault("operations", set()).add(ops)

    def print_all(self, op):
        #        print(" ".join(sorted(self.keys_["funcs"])))
        funcs_hdr = []

        nthr_hdr = []
        for nthread in sorted(self.keys_["nthreads"]):
            nthr_hdr.append(str(nthread))

            func_hdr_local = []
            for func in sorted(self.keys_["funcs"]):
                func_hdr_local.append(str(func).replace("memcpy_", ""))
            funcs_hdr.append(",".join(func_hdr_local) + ",")

        funcs_hdr = "impl=," + ",".join(funcs_hdr).upper()
        nthr_hdr = "nthreads=," + ",,,,".join(nthr_hdr)
        print(funcs_hdr)
        print(nthr_hdr)

        operation_strs = {
            0: "Standard",
            1: "PingPong",
            2: "Forward",
            3: "Forward",
            4: "Read",
            5: "Read-Dep"
        }
        for operation in sorted(self.keys_["operations"]):
            if operation != int(op):
                continue
            print("\nOperation={}/".format(operation,
                                           operation_strs[int(operation)]))
            for alignment in sorted(self.keys_["alignments"]):
                print("\nAlign={}".format(alignment))
                for size in sorted(self.keys_["sizes"]):
                    print("Size={},".format(size), end="")
                    for nthread in sorted(self.keys_["nthreads"]):
                        times = []
                        for func in sorted(self.keys_["funcs"]):
                            key = item_key(func, nthread, 0, size, alignment,
                                           operation)
                            assert key in self.results_
                            times.append(self.results_[key].get_stat())
                        avg = statistics.mean(times)
                        for i in range(0, len(times)):
                            times[i] = str(round(times[i] / avg, 3))
                        print(",".join(times) + ",", end=",")
                    print("")


res = Results()

lc = 0
for line in open(sys.argv[1]):
    if (lc & ((1 << 17) - 1)) == 0:
        print(lc)
    res.add_line(line)
    lc += 1

res.print_all(sys.argv[2])
