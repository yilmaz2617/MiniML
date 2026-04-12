#pragma once
#include "math_module.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace miniml {

// ─────────────────────────────────────────────────
// Optimizer seçimi
// ─────────────────────────────────────────────────
enum class Optimizer { SGD, Adam };

// ─────────────────────────────────────────────────
// Fully-Connected (Dense) Katman — SGD veya Adam
// ─────────────────────────────────────────────────
struct Layer {
    Matrix weights, biases;
    activations::Activation act;
    Matrix input, z, output;

    // Adam optimizer state (lazy init — ilk backward_adam çağrısında doldurulur)
    Matrix m_w, v_w, m_b, v_b;
    int adam_t = 0;

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

    // Vanilla SGD güncelleme
    Matrix backward(const Matrix& dOut, double lr) {
        Matrix dZ     = dOut.hadamard(z.apply(act.dfn));
        Matrix dW     = input.T() * dZ * (1.0 / (double)dZ.rows);
        Matrix dB     = dZ.col_sum() * (1.0 / (double)dZ.rows);
        Matrix dInput = dZ * weights.T();
        weights = weights - (dW * lr);
        biases  = biases  - (dB * lr);
        return dInput;
    }

    // Adam optimizer güncelleme (β1=0.9, β2=0.999)
    // Avantajı: adaptive LR, sparse gradient desteği, daha hızlı yakınsama
    Matrix backward_adam(const Matrix& dOut, double lr,
                         double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) {
        // Lazy init
        if (m_w.data.empty()) {
            m_w = Matrix(weights.rows, weights.cols, 0.0);
            v_w = Matrix(weights.rows, weights.cols, 0.0);
            m_b = Matrix(biases.rows,  biases.cols,  0.0);
            v_b = Matrix(biases.rows,  biases.cols,  0.0);
        }
        ++adam_t;

        Matrix dZ     = dOut.hadamard(z.apply(act.dfn));
        Matrix dW     = input.T() * dZ * (1.0 / (double)dZ.rows);
        Matrix dB     = dZ.col_sum() * (1.0 / (double)dZ.rows);
        Matrix dInput = dZ * weights.T();

        double bc1 = 1.0 - std::pow(beta1, adam_t);
        double bc2 = 1.0 - std::pow(beta2, adam_t);

        // Ağırlık güncelleme
        for (size_t i = 0; i < weights.data.size(); ++i) {
            m_w.data[i] = beta1 * m_w.data[i] + (1 - beta1) * dW.data[i];
            v_w.data[i] = beta2 * v_w.data[i] + (1 - beta2) * dW.data[i] * dW.data[i];
            weights.data[i] -= lr * (m_w.data[i] / bc1) / (std::sqrt(v_w.data[i] / bc2) + eps);
        }
        // Bias güncelleme
        for (size_t i = 0; i < biases.data.size(); ++i) {
            m_b.data[i] = beta1 * m_b.data[i] + (1 - beta1) * dB.data[i];
            v_b.data[i] = beta2 * v_b.data[i] + (1 - beta2) * dB.data[i] * dB.data[i];
            biases.data[i] -= lr * (m_b.data[i] / bc1) / (std::sqrt(v_b.data[i] / bc2) + eps);
        }
        return dInput;
    }
};

// ─────────────────────────────────────────────────
// Batch Normalization Katmanı
// Eğitim kararlılığını ve hızını artırır; internal covariate shift önler
// ─────────────────────────────────────────────────
struct BatchNormLayer {
    size_t features;
    Matrix gamma, beta;           // öğrenilebilir scale & shift
    Matrix running_mean, running_var;
    double momentum, eps;
    bool training;

    // İleri geçiş cache'i (backward için)
    Matrix cache_xhat, cache_input, cache_mean, cache_var;

    BatchNormLayer(size_t f, double mom = 0.1, double e = 1e-5)
        : features(f), momentum(mom), eps(e), training(true) {
        gamma        = Matrix(1, f, 1.0);
        beta         = Matrix(1, f, 0.0);
        running_mean = Matrix(1, f, 0.0);
        running_var  = Matrix(1, f, 1.0);
    }

    Matrix forward(const Matrix& x) {
        cache_input = x;
        size_t N = x.rows;
        Matrix xhat(N, features);

        if (training) {
            cache_mean = x.col_mean();
            cache_var  = x.col_var(cache_mean);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < features; ++j)
                    xhat(i, j) = (x(i, j) - cache_mean(0, j))
                                 / std::sqrt(cache_var(0, j) + eps);
            // Running stat güncelleme
            for (size_t j = 0; j < features; ++j) {
                running_mean(0, j) = (1 - momentum) * running_mean(0, j)
                                     + momentum * cache_mean(0, j);
                running_var(0, j)  = (1 - momentum) * running_var(0, j)
                                     + momentum * cache_var(0, j);
            }
        } else {
            // Inference: running stats kullan
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < features; ++j)
                    xhat(i, j) = (x(i, j) - running_mean(0, j))
                                 / std::sqrt(running_var(0, j) + eps);
        }
        cache_xhat = xhat;

        // y = γ * xhat + β
        Matrix out(N, features);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < features; ++j)
                out(i, j) = gamma(0, j) * xhat(i, j) + beta(0, j);
        return out;
    }

    Matrix backward(const Matrix& dout, double lr) {
        size_t N = cache_input.rows;

        Matrix dgamma(1, features, 0.0), dbeta(1, features, 0.0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < features; ++j) {
                dgamma(0, j) += dout(i, j) * cache_xhat(i, j);
                dbeta(0, j)  += dout(i, j);
            }

        // dx_hat = dout * γ
        Matrix dxhat(N, features);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < features; ++j)
                dxhat(i, j) = dout(i, j) * gamma(0, j);

        // Tam BN backward (Ioffe & Szegedy 2015)
        Matrix dx(N, features, 0.0);
        for (size_t j = 0; j < features; ++j) {
            double inv_sv = 1.0 / std::sqrt(cache_var(0, j) + eps);
            double sum_d  = 0.0, sum_dx = 0.0;
            for (size_t i = 0; i < N; ++i) {
                sum_d  += dxhat(i, j);
                sum_dx += dxhat(i, j) * cache_xhat(i, j);
            }
            for (size_t i = 0; i < N; ++i)
                dx(i, j) = inv_sv * (dxhat(i, j)
                            - sum_d / (double)N
                            - cache_xhat(i, j) * sum_dx / (double)N);
        }

        gamma = gamma - (dgamma * lr);
        beta  = beta  - (dbeta  * lr);
        return dx;
    }
};

// ─────────────────────────────────────────────────
// 2D Konvolüsyon Katmanı
// Input/Output: (N, C*H*W) düzleştirilmiş format
// conv.forward(x, H, W) şeklinde çağrılır
// ─────────────────────────────────────────────────
struct Conv2DLayer {
    int in_c, out_c, kh, kw, stride, pad;
    Matrix filters;   // (out_c, in_c*kh*kw)
    Matrix bias;      // (1, out_c)
    Matrix cached_input;
    int cached_N = 0, cached_H = 0, cached_W = 0;

    Conv2DLayer(int ic, int oc, int k = 3, int s = 1, int p = 0)
        : in_c(ic), out_c(oc), kh(k), kw(k), stride(s), pad(p) {
        filters = Matrix(out_c, ic * k * k);
        filters.xavier_init(ic * k * k, out_c);
        bias = Matrix(1, out_c, 0.0);
    }

    // out_H = (H + 2*pad - kh) / stride + 1
    int out_size(int sz, int k) const { return (sz + 2*pad - k) / stride + 1; }

    // İleri geçiş: doğrudan konvolüsyon (eğitsel, saf C++)
    Matrix forward(const Matrix& x, int H, int W) {
        int N = (int)x.rows;
        int oH = out_size(H, kh), oW = out_size(W, kw);
        cached_N = N; cached_H = H; cached_W = W;
        cached_input = x;

        Matrix out(N, out_c * oH * oW, 0.0);
        for (int n = 0; n < N; ++n)
          for (int oc = 0; oc < out_c; ++oc)
            for (int oh = 0; oh < oH; ++oh)
              for (int ow = 0; ow < oW; ++ow) {
                double val = bias(0, oc);
                for (int ic = 0; ic < in_c; ++ic)
                  for (int ki = 0; ki < kh; ++ki)
                    for (int kj = 0; kj < kw; ++kj) {
                        int ih = oh * stride - pad + ki;
                        int iw = ow * stride - pad + kj;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            val += x(n, ic*H*W + ih*W + iw)
                                 * filters(oc, ic*kh*kw + ki*kw + kj);
                    }
                out(n, oc*oH*oW + oh*oW + ow) = val;
              }
        return out;
    }

    // Geri yayılım: dfilters, dbias, dinput
    Matrix backward(const Matrix& dout, double lr) {
        int N = cached_N, H = cached_H, W = cached_W;
        int oH = out_size(H, kh), oW = out_size(W, kw);

        Matrix dfilters(out_c, in_c*kh*kw, 0.0);
        Matrix dbias(1, out_c, 0.0);
        Matrix dinput(N, in_c*H*W, 0.0);

        for (int n = 0; n < N; ++n)
          for (int oc = 0; oc < out_c; ++oc)
            for (int oh = 0; oh < oH; ++oh)
              for (int ow = 0; ow < oW; ++ow) {
                double d = dout(n, oc*oH*oW + oh*oW + ow);
                dbias(0, oc) += d;
                for (int ic = 0; ic < in_c; ++ic)
                  for (int ki = 0; ki < kh; ++ki)
                    for (int kj = 0; kj < kw; ++kj) {
                        int ih = oh*stride - pad + ki;
                        int iw = ow*stride - pad + kj;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            dfilters(oc, ic*kh*kw+ki*kw+kj) +=
                                cached_input(n, ic*H*W+ih*W+iw) * d;
                            dinput(n, ic*H*W+ih*W+iw) +=
                                filters(oc, ic*kh*kw+ki*kw+kj) * d;
                        }
                    }
              }

        double inv_N = 1.0 / N;
        filters = filters - (dfilters * (lr * inv_N));
        bias    = bias    - (dbias    * (lr * inv_N));
        return dinput;
    }
};

// ─────────────────────────────────────────────────
// 2D Max Pooling Katmanı
// ─────────────────────────────────────────────────
struct MaxPool2DLayer {
    int ph, pw, stride;
    Matrix cached_input;
    std::vector<int> argmax;   // backward için max konumları
    int cached_N = 0, cached_C = 0, cached_H = 0, cached_W = 0;

    MaxPool2DLayer(int pool_h = 2, int pool_w = 2, int s = 2)
        : ph(pool_h), pw(pool_w), stride(s) {}

    Matrix forward(const Matrix& x, int C, int H, int W) {
        int N = (int)x.rows;
        int oH = (H - ph) / stride + 1;
        int oW = (W - pw) / stride + 1;
        cached_N = N; cached_C = C; cached_H = H; cached_W = W;
        cached_input = x;
        argmax.assign(N * C * oH * oW, 0);

        Matrix out(N, C * oH * oW, -1e18);
        for (int n = 0; n < N; ++n)
          for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < oH; ++oh)
              for (int ow = 0; ow < oW; ++ow) {
                double best = -1e18; int best_idx = 0;
                for (int pi = 0; pi < ph; ++pi)
                  for (int pj = 0; pj < pw; ++pj) {
                    int ih = oh*stride + pi, iw = ow*stride + pj;
                    int flat = c*H*W + ih*W + iw;
                    if (x(n, flat) > best) { best = x(n, flat); best_idx = flat; }
                  }
                int oidx = c*oH*oW + oh*oW + ow;
                out(n, oidx) = best;
                argmax[n*C*oH*oW + oidx] = best_idx;
              }
        return out;
    }

    Matrix backward(const Matrix& dout) {
        int oH = (cached_H - ph) / stride + 1;
        int oW = (cached_W - pw) / stride + 1;
        Matrix dinput(cached_N, cached_C*cached_H*cached_W, 0.0);
        for (int n = 0; n < cached_N; ++n)
          for (int c = 0; c < cached_C; ++c)
            for (int oh = 0; oh < oH; ++oh)
              for (int ow = 0; ow < oW; ++ow) {
                int oidx = c*oH*oW + oh*oW + ow;
                dinput(n, argmax[n*cached_C*oH*oW + oidx]) += dout(n, oidx);
              }
        return dinput;
    }
};

// ─────────────────────────────────────────────────
// Flatten Katmanı
// CNN çıkışını Dense katmana bağlar
// ─────────────────────────────────────────────────
struct FlattenLayer {
    size_t cached_rows = 0, cached_cols = 0;

    Matrix forward(const Matrix& x) {
        cached_rows = x.rows; cached_cols = x.cols;
        return x;  // Zaten (N, flat) formatında
    }

    Matrix backward(const Matrix& dout) {
        return dout;  // Şekil değiştirmez, gradyan olduğu gibi geçer
    }
};

// ─────────────────────────────────────────────────
// Loss Fonksiyonları
// ─────────────────────────────────────────────────
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
            g.data[i] = (-target.data[i] / p + (1.0 - target.data[i]) / (1.0 - p))
                        / (double)pred.data.size();
        }
        return g;
    }
    // Huber: |e|≤δ → MSE, dışı → linear (gradient clipping etkisi)
    inline Matrix huber_grad(const Matrix& pred, const Matrix& target, double delta = 1.0) {
        Matrix g(pred.rows, pred.cols);
        double n = (double)pred.data.size();
        for (size_t i = 0; i < pred.data.size(); ++i) {
            double e = pred.data[i] - target.data[i];
            g.data[i] = (std::abs(e) <= delta ? e : delta * (e > 0 ? 1.0 : -1.0)) / n;
        }
        return g;
    }
    // DQN: çift normalizasyon önlemek için ham fark — Layer::backward (1/N) yeter
    inline Matrix dqn_grad(const Matrix& q_current, const Matrix& q_target) {
        return q_current - q_target;
    }
}

// ─────────────────────────────────────────────────
// NeuralNet — Fully-Connected ağ (SGD veya Adam)
// ─────────────────────────────────────────────────
class NeuralNet {
public:
    std::vector<Layer> layers;
    std::vector<double> loss_history;
    Optimizer optimizer = Optimizer::SGD;

    void add(size_t in, size_t out, const std::string& activation = "relu") {
        layers.emplace_back(in, out, activation);
    }

    void use_adam() { optimizer = Optimizer::Adam; }

    Matrix forward(const Matrix& x) {
        Matrix out = x;
        for (auto& layer : layers) out = layer.forward(out);
        return out;
    }

    void backward(const Matrix& loss_grad, double lr) {
        Matrix grad = loss_grad;
        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            if (optimizer == Optimizer::Adam)
                grad = layers[i].backward_adam(grad, lr);
            else
                grad = layers[i].backward(grad, lr);
        }
    }

    // CNN gibi alt katmanlarla zincir kurmak için input gradyanı döndürür
    Matrix backward_with_grad(const Matrix& loss_grad, double lr) {
        Matrix grad = loss_grad;
        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            if (optimizer == Optimizer::Adam)
                grad = layers[i].backward_adam(grad, lr);
            else
                grad = layers[i].backward(grad, lr);
        }
        return grad;
    }

    void train(const Matrix& X, const Matrix& Y, int epochs, double lr,
               const std::string& loss_fn = "mse", int print_every = 100) {
        printf("=== Training Start [%s] ===\n",
               optimizer == Optimizer::Adam ? "Adam" : "SGD");
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int e = 1; e <= epochs; ++e) {
            Matrix pred = forward(X);
            double l; Matrix grad;
            if (loss_fn == "bce") { l = loss::bce(pred, Y); grad = loss::bce_grad(pred, Y); }
            else                  { l = loss::mse(pred, Y); grad = loss::mse_grad(pred, Y); }
            loss_history.push_back(l);
            backward(grad, lr);
            if (e % print_every == 0 || e == 1)
                printf("  Epoch %5d | Loss: %.6f\n", e, l);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("=== Training Done === (%.1f ms)\n",
               std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double accuracy(const Matrix& X, const Matrix& Y, double threshold = 0.5) {
        Matrix pred = forward(X);
        int correct = 0;
        for (size_t i = 0; i < pred.rows; ++i) {
            if (Y.cols == 1) {
                if ((pred(i, 0) >= threshold ? 1 : 0) == (int)Y(i, 0)) correct++;
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
        f.close(); printf("Loss saved to %s\n", path.c_str());
    }

    void save(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        size_t n = layers.size();
        f.write((char*)&n, sizeof(n));
        for (auto& layer : layers) {
            f.write((char*)&layer.weights.rows, sizeof(size_t));
            f.write((char*)&layer.weights.cols, sizeof(size_t));
            f.write((char*)layer.weights.data.data(), layer.weights.data.size() * sizeof(double));
            f.write((char*)layer.biases.data.data(),  layer.biases.data.size()  * sizeof(double));
        }
        f.close(); printf("Model saved to %s\n", path.c_str());
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        size_t n; f.read((char*)&n, sizeof(n));
        for (size_t i = 0; i < n && i < layers.size(); ++i) {
            size_t r, c;
            f.read((char*)&r, sizeof(size_t));
            f.read((char*)&c, sizeof(size_t));
            f.read((char*)layers[i].weights.data.data(), r * c * sizeof(double));
            f.read((char*)layers[i].biases.data.data(),  c * sizeof(double));
        }
        f.close(); printf("Model loaded from %s\n", path.c_str());
    }
};

} // namespace miniml
