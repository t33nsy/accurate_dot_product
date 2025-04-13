#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

/***
 * Accurate sum function. Error-free transformation
 * @param a - first double number
 * @param b - second double number
 * @param sum - sum of a and b
 * @param err - error of transformation
 */
void accurateSum(const double &a, const double &b, double &sum, double &err) {
    sum = a + b;             // x = fl(a + b)
    double vB = sum - a;     // z = fl(x - a)
    double vA = sum - vB;    // fl(x - z)
    double deltaB = b - vB;  // fl(b - z)
    double deltaA = a - vA;  // fl(a - (x - z))
    err = deltaA + deltaB;   // y = fl((a - (x - z)) + (b - z))
}

/***
 * Veltkamp function [x, y] = TwoProduct(a, b)
 * Accurate prod function. Error-free transformation
 * @param a - first double number
 * @param b - second double number
 * @param prod - product of a and b
 * @param err - error of transformation
 */
void accurateProd(const double &a, const double &b, double &prod, double &err) {
    prod = a * b;
// if Fused-Multiply-and-Add (FMA) is available
#if defined(__FMA__) || defined(__FMA)
    // FMA(a, b, c) = (a * b + c)/(1 + epsilon1) + eta1
    err = std::fma(a, b, -prod);
#else
    // Dekker's method (splitting)
    const double split = 134217729.0;  // factor = fl(2^s+1) [2^27 + 1]
    double aHigh, aLow, bHigh, bLow;   // a1, a2, b1, b2
    double aSplit = a * split;         // c = fl(factor * a)
    aHigh = aSplit - (aSplit - a);     // a1 = fl(c - (c - a))
    aLow = a - aHigh;                  // a2 = fl(a - a1)
    double bSplit = b * split;         // c = fl(factor * b)
    bHigh = bSplit - (bSplit - b);     // b1 = fl(c - (c - b))
    bLow = b - bHigh;                  // b2 = fl(b - b1)
    err = ((aHigh * bHigh - prod) + aHigh * bLow + aLow * bHigh) + aLow * bLow;
#endif
}

/***
 * Accurate N numbers sum
 * Shewchuck-style algorithm
 * @param values - vector of numbers
 */
double compensatedSum(const std::vector<double> &values) {
    std::vector<double> partials;
    for (double x : values) {
        size_t i = 0;
        for (; i < partials.size(); ++i) {
            double high, low;
            accurateSum(partials[i], x, high, low);
            if (low != 0.0) {
                x = low;
                partials[i] = high;
            } else {
                x = high;
                partials[i] = 0.0;
            }
        }
        partials.push_back(x);
    }
    double sum = 0.0;
    for (double p : partials) {
        sum += p;
    }
    return sum;
}

/***
 * Dot product function
 * @param x - first vector of numbers
 * @param y - second vector of numbers
 * @return double - accurate dot product of x and y
 * @throws std::invalid_argument if x and y are not the same size
 */
double compensatedDotProduct(const std::vector<double> &x,
                             const std::vector<double> &y) {
    size_t n = x.size(), m = y.size();
    if (n != m) {
        throw std::invalid_argument("Vectors must be the same size");
    }
    // First step - collecting all prods and errors
    std::vector<double> partials;
    for (size_t i = 0; i < n; ++i) {
        double prod, err;
        accurateProd(x[i], y[i], prod, err);
        // adding to partials list
        partials.push_back(prod);
        partials.push_back(err);
    }
    // Second step - summation
    return compensatedSum(partials);
}

/***
 * Bit comparison
 * @param a - first double
 * @param b - second double
 * @return bool
 */
bool bitComparison(double a, double b) {
    uint64_t ua, ub;
    std::memcpy(&ua, &a, sizeof(double));
    std::memcpy(&ub, &b, sizeof(double));
    return ua == ub;
}

/***
 * Test invariant for shuffling
 * @param x - first vector of numbers
 * @param y - second vector of numbers
 */
void shuffleInvariant(const std::vector<double> &x,
                      const std::vector<double> &y) {
    double ref = compensatedDotProduct(x, y);
    std::vector<size_t> idx(x.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);
    std::vector<double> shuffled_x(x.size()), shuffled_y(y.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        shuffled_x[i] = x[idx[i]];
        shuffled_y[i] = y[idx[i]];
    }
    double sum = compensatedDotProduct(shuffled_x, shuffled_y);
    if (bitComparison(ref, sum)) {
        std::cout << "Test [Shuffle] passed\n";
    } else {
        std::cout << "Test [Shuffle] failed!\n";
        std::cout << "Expected: " << sum << ", Actual: " << ref << '\n';
    }
}

/***
 * Assert function for unit tests
 * @param expected
 * @param actual
 * @param label
 */
void AssertEqual(double expected, double actual, std::string label) {
    if (bitComparison(expected, actual)) {
        std::cout << "Test [" << label << "] passed\n";
    } else {
        std::cout << "Test [" << label << "] failed!\n";
        std::cout << "Expected: " << expected << ", Actual: " << actual << '\n';
    }
}

/***
 * Run unit tests for compensatedDotProduct
 */
void runUnitTests() {
    {
        double a = 0.123456789;
        double b = 0.987654321;
        double expected = 0.121932631112635269;
        double prod, err;
        accurateProd(a, b, prod, err);
        std::cout << "TEST [Simple Prod]\n"
                  << "Prod: " << prod << " Err: " << err << " Actual "
                  << expected << '\n';
        accurateProd(b, a, prod, err);
        std::cout << "TEST [Simple Prod Inverted]\n"
                  << "Prod: " << prod << " Err: " << err << " Actual "
                  << expected << '\n';
    }

    {
        double a = 0.123456789;
        double b = 0.987654321;
        double expected = 1.11111111;
        double sum, err;
        accurateSum(a, b, sum, err);
        std::cout << "TEST [Simple Sum]\n"
                  << "Sum: " << sum << " Err: " << err << " Actual " << expected
                  << '\n';
        accurateSum(b, a, sum, err);
        std::cout << "TEST [Simple Sum Inverted]\n"
                  << "Sum: " << sum << " Err: " << err << " Actual " << expected
                  << '\n';
    }

    {
        std::vector<double> arr = {0.123456789, 0.987654321, 0.1245153,
                                   12.514121, -1.51351};
        double expected = 12.23623741;
        double sum = compensatedSum(arr);
        AssertEqual(sum, expected, "Simple array sum");
    }

    {
        std::vector<double> x = {1.0, 2.0, 3.0};
        std::vector<double> y = {4.0, 5.0, 6.0};
        double expected = 32.0;
        AssertEqual(expected, compensatedDotProduct(x, y),
                    "Simple dot product test");
    }

    {
        std::vector<double> x = {0.0, 0.0, 0.0};
        std::vector<double> y = {1.0, 2.0, 3.0};
        AssertEqual(0.0, compensatedDotProduct(x, y), "Zero vector");
    }

    {
        std::vector<double> x = {1.0, 1e100, 1.0, -1e100};
        std::vector<double> y = {1.0, 1.0, -1.0, 1.0};
        AssertEqual(0.0, compensatedDotProduct(x, y), "Cancellation case");
        shuffleInvariant(x, y);
    }

    {
        std::vector<double> x(10000, 1e-10);
        std::vector<double> y(10000, 1.0);
        double expected = 1e-10 * 10000;  // 1e-6
        AssertEqual(expected, compensatedDotProduct(x, y), "Small numbers sum");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(15);
        runUnitTests();
        return 0;
    }
    std::cout << "Input the size of the vectors\n";
    size_t n;
    std::cin >> n;
    std::cout << "Please input the vector of numbers to be multiplied:\n";
    std::vector<double> A(n), B(n);
    std::cout << "A: ";
    for (size_t i = 0; i < n; ++i) std::cin >> A[i];
    std::cout << "B: ";
    for (size_t i = 0; i < n; ++i) std::cin >> B[i];
    double sum = compensatedDotProduct(A, B);
    std::cout << "Dot product = " << sum;
    shuffleInvariant(A, B);
    return 0;
}