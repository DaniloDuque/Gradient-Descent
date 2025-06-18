#pragma once
#include "math/vector/Vector.h"
#include <initializer_list>

namespace math {

    class Matrix {
    public:
        Matrix(size_t rows, size_t cols, double init_val = 0.0);
        Matrix(std::initializer_list<Vector> init);
        Matrix() = default;

        Matrix transpose() const;

        double& operator()(size_t i, size_t j);
        double operator()(size_t i, size_t j) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(double scalar) const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Vector operator*(const Vector& vec) const;

        static Matrix identity(size_t n);
        static Matrix zeros(size_t rows, size_t cols);

    private:
        size_t m_rows{}, m_cols{};
        Vector m_data;
        const double EPSILON = 1e-5;
    };

}