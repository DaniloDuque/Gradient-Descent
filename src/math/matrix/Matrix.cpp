#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <cassert>

namespace math {
    Matrix::Matrix(const size_t rows, const size_t cols, const double init_val): m_rows(rows), m_cols(cols), m_data(rows * cols, init_val) {}

    Matrix::Matrix(const std::initializer_list<Vector> init) {
        m_rows = init.size();
        m_cols = init.begin()->size();
        m_data.reserve(m_rows * m_cols);

        for (const auto& row : init) {
            assert(row.size() == m_cols);
            m_data.insert(m_data.end(), row.begin(), row.end());
        }
    }

    double& Matrix::operator()(const size_t i, const size_t j) {
        assert(i < m_rows && j < m_cols);
        return m_data[i * m_cols + j];
    }

    double Matrix::operator()(const size_t i, const size_t j) const {
        assert(i < m_rows && j < m_cols);
        return m_data[i * m_cols + j];
    }
    
    Matrix Matrix::transpose() const {
        Matrix result(m_cols, m_rows);
        for (size_t i = 0; i < m_rows; ++i)
            for (size_t j = 0; j < m_cols; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }

    Matrix Matrix::operator*(const Matrix& other) const {
        assert(m_cols == other.m_rows);
        Matrix result(m_rows, other.m_cols);

        for (size_t i = 0; i < m_rows; ++i)
            for (size_t j = 0; j < other.m_cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < m_cols; ++k)
                    sum += (*this)(i, k) * other(k, j);
                result(i, j) = sum;
            }
        return result;
    }

    Vector Matrix::operator*(const Vector& vec) const {
        assert(m_cols == vec.size());
        Vector result(m_rows);

        for (size_t i = 0; i < m_rows; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < m_cols; ++j)
                sum += (*this)(i, j) * vec[j];
            result[i] = sum;
        }

        return result;
    }

    Matrix Matrix::operator*(const double scalar) const {
        Matrix result(m_rows, m_cols);
        result.m_data = m_data * scalar;
        return result;
    }

    Matrix Matrix::operator+(const Matrix& other) const {
        assert(m_rows == other.m_rows && m_cols == other.m_cols);
        Matrix result(m_rows, m_cols);
        result.m_data = m_data + other.m_data;
        return result;
    }

    Matrix Matrix::operator-(const Matrix& other) const {
        assert(m_rows == other.m_rows && m_cols == other.m_cols);
        Matrix result(m_rows, m_cols);
        result.m_data = m_data - other.m_data;
        return result;
    }

    Matrix Matrix::identity(const size_t n) {
        Matrix I(n, n);
        for (size_t i = 0; i < n; ++i) I(i, i) = 1.0;
        return I;
    }

    Matrix Matrix::zeros(const size_t rows, const size_t cols) {
        Matrix Z(rows, cols);
        return Z;
    }
    
}