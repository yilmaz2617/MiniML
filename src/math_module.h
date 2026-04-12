#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <functional>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace miniml {

class Matrix {
public:
    size_t rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(size_t r, size_t c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    double& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    double  operator()(size_t i, size_t j) const { return data[i * cols + j]; }

    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows);
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t k = 0; k < cols; ++k)
                for (size_t j = 0; j < other.cols; ++j)
                    result(i, j) += (*this)(i, k) * other(k, j);
        return result;
    }

    Matrix operator+(const Matrix& o) const {
        assert(rows == o.rows && cols == o.cols);
        Matrix r(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            r.data[i] = data[i] + o.data[i];
        return r;
    }

    Matrix operator-(const Matrix& o) const {
        assert(rows == o.rows && cols == o.cols);
        Matrix r(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            r.data[i] = data[i] - o.data[i];
        return r;
    }

    Matrix hadamard(const Matrix& o) const {
        assert(rows == o.rows && cols == o.cols);
        Matrix r(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            r.data[i] = data[i] * o.data[i];
        return r;
    }

    Matrix operator*(double s) const {
        Matrix r(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            r.data[i] = data[i] * s;
        return r;
    }

    Matrix T() const {
        Matrix r(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                r(j, i) = (*this)(i, j);
        return r;
    }

    Matrix add_bias(const Matrix& bias) const {
        assert(bias.rows == 1 && bias.cols == cols);
        Matrix r = *this;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                r(i, j) += bias(0, j);
        return r;
    }

    Matrix apply(std::function<double(double)> fn) const {
        Matrix r(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            r.data[i] = fn(data[i]);
        return r;
    }

    double sum() const {
        return std::accumulate(data.begin(), data.end(), 0.0);
    }

    Matrix col_sum() const {
        Matrix r(1, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                r(0, j) += (*this)(i, j);
        return r;
    }

    Matrix col_mean() const {
        Matrix r = col_sum();
        for (size_t j = 0; j < cols; ++j) r(0, j) /= (double)rows;
        return r;
    }

    Matrix col_var(const Matrix& mean) const {
        assert(mean.rows == 1 && mean.cols == cols);
        Matrix r(1, cols, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                double d = (*this)(i, j) - mean(0, j);
                r(0, j) += d * d;
            }
        for (size_t j = 0; j < cols; ++j) r(0, j) /= (double)rows;
        return r;
    }

    void xavier_init(size_t fan_in, size_t fan_out) {
        std::mt19937 gen(42);
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (auto& v : data) v = dist(gen);
    }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " (" << rows << "x" << cols << "):\n";
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j)
                printf("%8.4f ", (*this)(i, j));
            std::cout << "\n";
        }
    }
};

namespace activations {

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoid_d(double x) { double s = sigmoid(x); return s * (1.0 - s); }

inline double relu(double x) { return x > 0 ? x : 0; }
inline double relu_d(double x) { return x > 0 ? 1.0 : 0.0; }

inline double tanh_act(double x) { return std::tanh(x); }
inline double tanh_d(double x) { double t = std::tanh(x); return 1.0 - t * t; }

struct Activation {
    std::string name;
    std::function<double(double)> fn;
    std::function<double(double)> dfn;
};

inline Activation get(const std::string& name) {
    if (name == "sigmoid") return {"sigmoid", sigmoid, sigmoid_d};
    if (name == "relu")    return {"relu", relu, relu_d};
    if (name == "tanh")    return {"tanh", tanh_act, tanh_d};
    return {"linear", [](double x){return x;}, [](double){return 1.0;}};
}

}
}