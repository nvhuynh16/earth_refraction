#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <sstream>

namespace refraction {
namespace test {

struct TestStats {
    int passed = 0;
    int failed = 0;
    int total = 0;
    std::vector<std::string> failures;
};

inline TestStats& global_stats() {
    static TestStats stats;
    return stats;
}

#ifdef _WIN32
    inline constexpr const char* GREEN = "";
    inline constexpr const char* RED = "";
    inline constexpr const char* YELLOW = "";
    inline constexpr const char* RESET = "";
#else
    inline constexpr const char* GREEN = "\033[32m";
    inline constexpr const char* RED = "\033[31m";
    inline constexpr const char* YELLOW = "\033[33m";
    inline constexpr const char* RESET = "\033[0m";
#endif

#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ << " - Assertion failed: " << #condition; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_MSG(condition, msg) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ << " - " << msg; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(actual, expected, tolerance) \
    do { \
        double _a = static_cast<double>(actual); \
        double _e = static_cast<double>(expected); \
        double _t = static_cast<double>(tolerance); \
        if (std::abs(_a - _e) > _t) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Expected " << std::setprecision(15) << _e \
                << " +/- " << _t << ", got " << _a \
                << " (diff=" << std::abs(_a - _e) << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_NEAR_REL(actual, expected, rel_tolerance) \
    do { \
        double _a = static_cast<double>(actual); \
        double _e = static_cast<double>(expected); \
        double _t = std::abs(_e * rel_tolerance); \
        if (_t < 1e-15) _t = 1e-15; \
        if (std::abs(_a - _e) > _t) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Expected " << std::setprecision(15) << _e \
                << " +/- " << (rel_tolerance * 100) << "%, got " << _a \
                << " (rel_diff=" << std::abs((_a - _e) / _e) * 100 << "%)"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_TRUE(condition) TEST_ASSERT(condition)
#define TEST_ASSERT_FALSE(condition) TEST_ASSERT(!(condition))

#define TEST_ASSERT_EQ(actual, expected) \
    do { \
        auto _a = (actual); \
        auto _e = (expected); \
        if (_a != _e) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Expected " << _e << ", got " << _a; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

class TestRunner {
public:
    using TestFunc = std::function<void()>;

    TestRunner(const std::string& suite_name) : suite_name_(suite_name) {
        std::cout << "\n=== " << suite_name_ << " ===" << std::endl;
    }

    void add_test(const std::string& name, TestFunc func) {
        tests_.push_back({name, func});
    }

    int run() {
        int passed = 0;
        int failed = 0;

        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& test : tests_) {
            std::cout << "  " << test.name << "... " << std::flush;
            try {
                test.func();
                std::cout << GREEN << "PASSED" << RESET << std::endl;
                passed++;
            } catch (const std::exception& e) {
                std::cout << RED << "FAILED" << RESET << std::endl;
                std::cout << "    " << e.what() << std::endl;
                failed++;
                global_stats().failures.push_back(suite_name_ + "::" + test.name);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\n  " << suite_name_ << " Summary: "
                  << passed << "/" << (passed + failed) << " passed"
                  << " (" << duration.count() << "ms)" << std::endl;

        global_stats().passed += passed;
        global_stats().failed += failed;
        global_stats().total += passed + failed;

        return failed;
    }

private:
    struct Test {
        std::string name;
        TestFunc func;
    };

    std::string suite_name_;
    std::vector<Test> tests_;
};

#define TEST_CASE(runner, test_name) \
    void test_##test_name(); \
    struct TestRegistrar_##test_name { \
        TestRegistrar_##test_name() { runner.add_test(#test_name, test_##test_name); } \
    } registrar_##test_name; \
    void test_##test_name()

inline int print_final_summary() {
    auto& stats = global_stats();
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total: " << stats.total << " tests" << std::endl;
    std::cout << GREEN << "Passed: " << stats.passed << RESET << std::endl;
    if (stats.failed > 0) {
        std::cout << RED << "Failed: " << stats.failed << RESET << std::endl;
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& f : stats.failures) {
            std::cout << "  - " << f << std::endl;
        }
    }
    std::cout << "========================================" << std::endl;
    return stats.failed > 0 ? 1 : 0;
}

}  // namespace test
}  // namespace refraction
