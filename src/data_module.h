#pragma once
#include "math_module.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>

namespace miniml {
namespace data {

inline Matrix read_csv(const std::string& path, bool has_header = true) {
    std::ifstream file(path);
    if (!file.is_open()) { std::cerr << "Error: Cannot open " << path << std::endl; return Matrix(); }
    std::vector<std::vector<double>> rows;
    std::string line;
    if (has_header) std::getline(file, line);
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            try { row.push_back(std::stod(val)); } catch (...) { row.push_back(0.0); }
        }
        if (!row.empty()) rows.push_back(row);
    }
    if (rows.empty()) return Matrix();
    Matrix m(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            m(i, j) = rows[i][j];
    std::cout << "Loaded " << m.rows << "x" << m.cols << " from " << path << std::endl;
    return m;
}

inline Matrix normalize(const Matrix& m) {
    Matrix r(m.rows, m.cols);
    for (size_t j = 0; j < m.cols; ++j) {
        double mn = m(0, j), mx = m(0, j);
        for (size_t i = 1; i < m.rows; ++i) { mn = std::min(mn, m(i, j)); mx = std::max(mx, m(i, j)); }
        double range = mx - mn;
        if (range < 1e-10) range = 1.0;
        for (size_t i = 0; i < m.rows; ++i) r(i, j) = (m(i, j) - mn) / range;
    }
    return r;
}

inline void split_xy(const Matrix& data, Matrix& X, Matrix& Y, size_t output_cols = 1) {
    size_t in_cols = data.cols - output_cols;
    X = Matrix(data.rows, in_cols);
    Y = Matrix(data.rows, output_cols);
    for (size_t i = 0; i < data.rows; ++i) {
        for (size_t j = 0; j < in_cols; ++j) X(i, j) = data(i, j);
        for (size_t j = 0; j < output_cols; ++j) Y(i, j) = data(i, in_cols + j);
    }
}

}
}