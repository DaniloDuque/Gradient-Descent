#pragma once
#include <memory>
#include <vector>
#include <set>

namespace autodiff {
    class Operation;

    class Variable {
    public:
        explicit Variable(double value, bool requires_grad = false);
        Variable(const Variable& other) = default;
        Variable& operator=(const Variable& other) = default;
        Variable(Variable&& other) = default;
        Variable& operator=(Variable&& other) = default;

        [[nodiscard]] double value() const { return value_; }
        [[nodiscard]] double grad() const { return grad_; }
        [[nodiscard]] bool requires_grad() const { return requires_grad_; }

        void set_grad(const double grad) { grad_ = grad; }
        void zero_grad() { grad_ = 0.0; }

        void backward();

        Variable operator+(const Variable& other) const;
        Variable operator-(const Variable& other) const;
        Variable operator*(const Variable& other) const;
        Variable operator/(const Variable& other) const;
        Variable operator-() const;
        [[nodiscard]] Variable pow(double exponent) const;
        [[nodiscard]] Variable exp() const;
        [[nodiscard]] Variable log() const;
        [[nodiscard]] Variable sin() const;
        [[nodiscard]] Variable cos() const;
        [[nodiscard]] Variable tanh() const;
        [[nodiscard]] Variable relu() const;
        [[nodiscard]] Variable sigmoid() const;

        void print() const;

    private:
        double value_;
        double grad_;
        bool requires_grad_;
        std::shared_ptr<Operation> grad_fn_;

        Variable(double value, bool requires_grad, std::shared_ptr<Operation> grad_fn);
        void topological_sort(std::vector<Variable*>& sorted, std::set<Variable*>& visited) const;

        friend class AddOperation;

    };

}