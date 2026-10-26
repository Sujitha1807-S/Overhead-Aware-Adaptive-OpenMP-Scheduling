#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

using Clock = std::chrono::steady_clock;

enum class Workload {
    Uniform,
    Irregular,
    Increasing
};

std::uint64_t iteration_work(int i, Workload workload, int n) {
    std::uint64_t x =
        static_cast<std::uint64_t>(i) + 0x9e3779b97f4a7c15ULL;

    int rounds = 0;

    switch (workload) {
        case Workload::Uniform:
            rounds = 160;
            break;

        case Workload::Irregular:
            rounds = 20 + static_cast<int>(
                (x ^ (x >> 17) ^ (x >> 31)) % 281
            );
            break;

        case Workload::Increasing:
            rounds = 20 + static_cast<int>(
                (static_cast<std::uint64_t>(i) * 280ULL) /
                static_cast<std::uint64_t>(std::max(1, n - 1))
            );
            break;
    }

    std::uint64_t value = x;

    for (int r = 0; r < rounds; ++r) {
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        value *= 0x2545F4914F6CDD1DULL;
    }

    return value;
}

std::uint64_t execute_loop(
    Workload workload,
    int policy,
    int n,
    int chunk
) {
    std::uint64_t total = 0;

    if (policy == 0) {
        #pragma omp parallel for schedule(static) reduction(+:total)
        for (int i = 0; i < n; ++i)
            total += iteration_work(i, workload, n);
    }
    else if (policy == 1) {
        #pragma omp parallel for schedule(dynamic, chunk) reduction(+:total)
        for (int i = 0; i < n; ++i)
            total += iteration_work(i, workload, n);
    }
    else if (policy == 2) {
        #pragma omp parallel for schedule(guided, chunk) reduction(+:total)
        for (int i = 0; i < n; ++i)
            total += iteration_work(i, workload, n);
    }
    else {
        // LB4OMP AUTO option 16 is selected through OMP_SCHEDULE.
        #pragma omp parallel for schedule(runtime) reduction(+:total)
        for (int i = 0; i < n; ++i)
            total += iteration_work(i, workload, n);
    }

    return total;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " <uniform|irregular|increasing>"
            << " <static|dynamic|guided|adaptive>"
            << " <repetition>\n";
        return 1;
    }

    const std::string workloadArg = argv[1];
    const std::string policyArg = argv[2];

    Workload workload;

    if (workloadArg == "uniform")
        workload = Workload::Uniform;
    else if (workloadArg == "irregular")
        workload = Workload::Irregular;
    else if (workloadArg == "increasing")
        workload = Workload::Increasing;
    else
        return 1;

    int policy = -1;

    if (policyArg == "static")
        policy = 0;
    else if (policyArg == "dynamic")
        policy = 1;
    else if (policyArg == "guided")
        policy = 2;
    else if (policyArg == "adaptive")
        policy = 3;
    else
        return 1;

    const int repetition = std::atoi(argv[3]);

    if (repetition < 1)
        return 1;

    constexpr int N = 100000;
    constexpr int CHUNK = 16;
    constexpr int ROUNDS = 17;

    omp_set_dynamic(0);
    omp_set_num_threads(4);

    // Compute a serial reference outside the timed interval.
    std::uint64_t expected = 0;

    for (int i = 0; i < N; ++i)
        expected += iteration_work(i, workload, N);

    // Warm up OpenMP without executing the adaptive loop.
    std::uint64_t warmup = 0;

    #pragma omp parallel for schedule(static) reduction(+:warmup)
    for (int i = 0; i < 1000; ++i)
        warmup += static_cast<std::uint64_t>(i);

    (void)warmup;

    bool correct = true;
    std::uint64_t checksum = 0;

    const auto start = Clock::now();

    for (int round = 0; round < ROUNDS; ++round) {
        const std::uint64_t result =
            execute_loop(workload, policy, N, CHUNK);

        checksum = result;

        if (result != expected)
            correct = false;
    }

    const auto stop = Clock::now();

    const double elapsedMs =
        std::chrono::duration<double, std::milli>(
            stop - start
        ).count();

    // CSV output: one row per independent process.
    std::cout
        << workloadArg << ","
        << policyArg << ","
        << repetition << ","
        << N << ","
        << ROUNDS << ","
        << elapsedMs << ","
        << checksum << ","
        << (correct ? "PASS" : "FAIL")
        << "\n";

    return correct ? 0 : 2;
}
