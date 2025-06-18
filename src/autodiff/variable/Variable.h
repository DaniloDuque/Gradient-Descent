#pragma once
#include <memory>
#include <vector>
#include <set>

namespace autodiff {
    class Operation;

    class Variable final : public std::enable_shared_from_this<Variable> {
    public:
        static std::shared_ptr<Variable> create(double value, bool requires_grad = false);
        static std::shared_ptr<Variable> create(double value, bool requires_grad,
                                               std::shared_ptr<Operation> grad_fn);

        double value() const { return value_; }
        double grad() const { return grad_; }
        bool requires_grad() const { return requires_grad_; }

        void set_value(const double new_value) { value_ = new_value; }
        void set_grad(const double grad) { grad_ = grad; }
        void zero_grad() { grad_ = 0.0; }
        void add_grad(const double grad) { grad_ += grad; }

        void backward();

        std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& other);
        std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& other);
        std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& other);
        std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& other);
        std::shared_ptr<Variable> operator-();

        std::shared_ptr<Variable> exp();
        std::shared_ptr<Variable> log();
        std::shared_ptr<Variable> pow(const std::shared_ptr<Variable>& other);
        std::shared_ptr<Variable> tanh();
        std::shared_ptr<Variable> cos();
        std::shared_ptr<Variable> sin();

        void print() const;

        ~Variable() = default;

    private:
        double value_;
        double grad_;
        bool requires_grad_;
        std::shared_ptr<Operation> grad_fn_;

        explicit Variable(double value, bool requires_grad = false);
        Variable(double value, bool requires_grad, std::shared_ptr<Operation> grad_fn);

        Variable(const Variable& other) = delete;
        Variable& operator=(const Variable& other) = delete;
        Variable(Variable&& other) = delete;
        Variable& operator=(Variable&& other) = delete;

        void topological_sort(std::vector<std::shared_ptr<Variable>>& sorted, std::set<std::shared_ptr<Variable>>& visited);

        friend class AddOperation;
        friend class MultiplyOperation;
        friend class DivideOperation;
        friend class SubtractOperation;
        friend class NegativeOperation;
        friend class ExponentialOperation;
        friend class LogarithmOperation;
        friend class PowerOperation;
        friend class TanhOperation;
        friend class SineOperation;
        friend class CosineOperation;
    };

    std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);
    std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);
    std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& lhs);
    std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);
    std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);
    std::shared_ptr<Variable> pow(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);

    std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& lhs, double rhs);
    std::shared_ptr<Variable> operator+(double lhs, const std::shared_ptr<Variable>& rhs);

    std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& lhs, double rhs);
    std::shared_ptr<Variable> operator-(double lhs, const std::shared_ptr<Variable>& rhs);

    std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& lhs, double rhs);
    std::shared_ptr<Variable> operator*(double lhs, const std::shared_ptr<Variable>& rhs);

    std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& lhs, double rhs);
    std::shared_ptr<Variable> operator/(double lhs, const std::shared_ptr<Variable>& rhs);

    std::shared_ptr<Variable> pow(const std::shared_ptr<Variable>& lhs, double rhs);
    std::shared_ptr<Variable> pow(double lhs, const std::shared_ptr<Variable>& rhs);

}