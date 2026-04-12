#pragma once
#include "math_module.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

namespace miniml {

struct Layer {
    Matrix weights;
    Matrix biases;
    activations::Activation act;
    Matrix input, z, output;

    Layer(size_t in, size_t out, const std::string& activation = "relu") {
        weights = Matrix(in, out);
        weights.xavier_init(in, out);
        biases = Matrix(1, out, 0.0);
        act = activations::get(activation);
    }

    Matrix forward(const Matrix& x) {
        input = x;
        z = (x * weights).add_bias(biases);
        output = z.apply(act.fn);
        return output;
    }

    Matrix backward(const Matrix& dOut, double lr) {
        Matrix dZ = dOut.hadamard(z.apply(act.dfn));
        Matrix dW = input.T() * dZ * (1.0 / (double)dZ.rows);
        Matrix dB = dZ.col_sum() * (1.0 / (double)dZ.rows);
        Matrix dInput = dZ * weights.T();
        weights = weights - (dW * lr);
        biases = biases - (dB * lr);
        return dInput;
    }
};

namespace loss {
    inline double mse(const Matrix& pred, const Matrix& target) {
        Matrix diff = pred - target;
        double s = 0;
        for (auto& v : diff.data) s += v * v;
        return s / (double)diff.data.size();
    }
    inline Matrix mse_grad(const Matrix& pred, const Matrix& target) {
        return (pred - target) * (2.0 / (double)pred.data.size());
    }
    inline double bce(const Matrix& pred, const Matrix& target) {
        double s = 0;
        for (size_t i = 0; i < pred.data.size(); ++i) {
            double p = std::clamp(pred.data[i], 1e-7, 1.0 - 1e-7);
            s += -(target.data[i] * std::log(p) + (1.0 - target.data[i]) * std::log(1.0 - p));
        }
        return s / (double)pred.data.size();
    }
    inline Matrix bce_grad(const Matrix& pred, const Matrix& target) {
        Matrix g(pred.rows, pred.cols);
        for (size_t i = 0; i < pred.data.size(); ++i) {
            double p = std::clamp(pred.data[i], 1e-7, 1.0 - 1e-7);
            g.data[i] = (-target.data[i] / p + (1.0 - target.data[i]) / (1.0 - p)) / (double)pred.data.size();
        }
        return g;
    }
}

class NeuralNet {
public:
    std::vector<Layer> layers;
    std::vector<double> loss_history;

    void add(size_t in, size_t out, const std::string& activation = "relu") {
        layers.emplace_back(in, out, activation);
    }

    Matrix forward(const Matrix& x) {
        Matrix out = x;
        for (auto& layer : layers) out = layer.forward(out);
        return out;
    }

    void backward(const Matrix& loss_grad, double lr) {
        Matrix grad = loss_grad;
        for (int i = (int)layers.size() - 1; i >= 0; --i)
            grad = layers[i].backward(grad, lr);
    }

    void train(const Matrix& X, const Matrix& Y, int epochs, double lr,
               const std::string& loss_fn = "mse", int print_every = 100) {
        std::cout << "=== Training Start ===" << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int e = 1; e <= epochs; ++e) {
            Matrix pred = forward(X);
            double l; Matrix grad;
            if (loss_fn == "bce") { l = loss::bce(pred, Y); grad = loss::bce_grad(pred, Y); }
            else { l = loss::mse(pred, Y); grad = loss::mse_grad(pred, Y); }
            loss_history.push_back(l);
            backward(grad, lr);
            if (e % print_every == 0 || e == 1)
                printf("  Epoch %5d | Loss: %.6f\n", e, l);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("=== Training Done === (%.1f ms)\n", ms);
    }

    double accuracy(const Matrix& X, const Matrix& Y, double threshold = 0.5) {
        Matrix pred = forward(X);
        int correct = 0;
        for (size_t i = 0; i < pred.rows; ++i) {
            if (Y.cols == 1) {
                int p = pred(i, 0) >= threshold ? 1 : 0;
                if (p == (int)Y(i, 0)) correct++;
            } else {
                size_t pi = 0, ti = 0;
                for (size_t j = 1; j < Y.cols; ++j) {
                    if (pred(i, j) > pred(i, pi)) pi = j;
                    if (Y(i, j) > Y(i, ti)) ti = j;
                }
                if (pi == ti) correct++;
            }
        }
        return (double)correct / (double)pred.rows * 100.0;
    }

    void save_loss(const std::string& path) {
        std::ofstream f(path); f << "epoch,loss\n";
        for (size_t i = 0; i < loss_history.size(); ++i)
            f << i + 1 << "," << loss_history[i] << "\n";
        f.close(); std::cout << "Loss saved to " << path << std::endl;
    }

    void save(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        size_t n = layers.size();
        f.write((char*)&n, sizeof(n));
        for (auto& layer : layers) {
            f.write((char*)&layer.weights.rows, sizeof(size_t));
            f.write((char*)&layer.weights.cols, sizeof(size_t));
            f.write((char*)layer.weights.data.data(), layer.weights.data.size() * sizeof(double));
            f.write((char*)layer.biases.data.data(), layer.biases.data.size() * sizeof(double));
        }
        f.close(); std::cout << "Model saved to " << path << std::endl;
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        size_t n; f.read((char*)&n, sizeof(n));
        for (size_t i = 0; i < n && i < layers.size(); ++i) {
            size_t r, c;
            f.read((char*)&r, sizeof(size_t));
            f.read((char*)&c, sizeof(size_t));
            f.read((char*)layers[i].weights.data.data(), r * c * sizeof(double));
            f.read((char*)layers[i].biases.data.data(), c * sizeof(double));
        }
        f.close(); std::cout << "Model loaded from " << path << std::endl;
    }
};

}