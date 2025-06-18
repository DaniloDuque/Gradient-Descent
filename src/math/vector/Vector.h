#pragma once
#include <vector>

namespace math {

    class Vector : public std::vector<double> {
    public:
        using std::vector<double>::vector;

        double norm() const;
        double dot(const Vector& other) const;

        Vector operator*(double scalar) const;
        Vector operator+(const Vector& other) const;
        Vector operator-(const Vector& other) const;

        void reserve(size_t size);
    };

}