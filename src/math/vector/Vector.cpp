#include "Vector.h"
#include <cassert>
#include <cmath>

namespace math {

    double Vector::norm() const {
        double result = 0.0;
        for (const auto& i : *this) result += i * i;
        return sqrt(result);
    }

    double Vector::dot(const Vector& other) const {
        double result = 0.0;
        for (size_t i = 0; i < size(); ++i) result += (*this)[i] * other[i];
        return result;
    }

    Vector Vector::operator*(const double scalar) const {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) result[i] = (*this)[i] * scalar;
        return result;
    }

    Vector Vector::operator+(const Vector& other) const {
        assert(size() == other.size());
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) result[i] = (*this)[i] + other[i];
        return result;
    }

    Vector Vector::operator-(const Vector& other) const {
        assert(size() == other.size());
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) result[i] = (*this)[i] - other[i];
        return result;
    }

    void Vector::reserve(const size_t size) {
        std::vector<double>::reserve(size);
    }
    
}