#include "Variable.h"
#include <iostream>
#include <cmath>
#include <ranges>
#include "autodiff/operation/Operation.h"
#include "autodiff/operation/arithmetic/AddOperation.cpp"
#include "autodiff/operation/arithmetic/SubtractOperation.cpp"
#include "autodiff/operation/arithmetic/MultiplyOperation.cpp"
#include "autodiff/operation/arithmetic/DivideOperation.cpp"
#include "autodiff/operation/arithmetic/NegativeOperation.cpp"
#include "autodiff/operation/expolog/LogarithmOperation.cpp"
#include "autodiff/operation/expolog/ExponentialOperation.cpp"
#include "autodiff/operation/expolog/PowerOperation.cpp"
#include "autodiff/operation/trigonometric/SineOperation.cpp"
#include "autodiff/operation/trigonometric/CosineOperation.cpp"
#include "autodiff/operation/hyperbolic/TanhOperation.cpp"

namespace autodiff {

    Variable::Variable(const double value, const bool requires_grad)
        : value_(value), grad_(0.0), requires_grad_(requires_grad), grad_fn_(nullptr) {}

    Variable::Variable(const double value, const bool requires_grad, std::shared_ptr<Operation> grad_fn)
        : value_(value), grad_(0.0), requires_grad_(requires_grad), grad_fn_(std::move(grad_fn)) {}

    std::shared_ptr<Variable> Variable::create(double value, bool requires_grad) {
        return std::shared_ptr<Variable>(new Variable(value, requires_grad));
    }

    std::shared_ptr<Variable> Variable::create(double value, bool requires_grad, std::shared_ptr<Operation> grad_fn) {
        return std::shared_ptr<Variable>(new Variable(value, requires_grad, std::move(grad_fn)));
    }

    void Variable::backward() {
        grad_ = 1.0;
        std::vector<std::shared_ptr<Variable>> sorted;
        std::set<std::shared_ptr<Variable>> visited;
        topological_sort(sorted, visited);

        for (const auto& it : std::ranges::reverse_view(sorted)) {
            if (it->grad_fn_) {
                it->grad_fn_->backward(it->grad_);
            }
        }
    }

    void Variable::topological_sort(std::vector<std::shared_ptr<Variable>>& sorted,
                                   std::set<std::shared_ptr<Variable>>& visited) {
        if (visited.contains(shared_from_this())) return;

        visited.insert(shared_from_this());

        if (grad_fn_) {
            for (const auto& input : grad_fn_->get_inputs()) {
                input->topological_sort(sorted, visited);
            }
        }

        sorted.push_back(shared_from_this());
    }

    std::shared_ptr<Variable> Variable::operator+(const std::shared_ptr<Variable>& other) {
        auto result = create(value_ + other->value_, requires_grad_ || other->requires_grad_);

        if (requires_grad_ || other->requires_grad_) {
            result->grad_fn_ = std::make_shared<AddOperation>(
                shared_from_this(),
                other
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::operator-(const std::shared_ptr<Variable>& other) {
        auto result = create(value_ - other->value_, requires_grad_ || other->requires_grad_);

        if (requires_grad_ || other->requires_grad_) {
            result->grad_fn_ = std::make_shared<SubtractOperation>(
                shared_from_this(),
                other
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::operator*(const std::shared_ptr<Variable>& other) {
        auto result = create(value_ * other->value_, requires_grad_ || other->requires_grad_);

        if (requires_grad_ || other->requires_grad_) {
            result->grad_fn_ = std::make_shared<MultiplyOperation>(
                shared_from_this(),
                other
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::operator/(const std::shared_ptr<Variable>& other) {
        auto result = create(value_ / other->value_, requires_grad_ || other->requires_grad_);

        if (requires_grad_ || other->requires_grad_) {
            result->grad_fn_ = std::make_shared<DivideOperation>(
                shared_from_this(),
                other
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::operator-() {
        auto result = create(-value_, requires_grad_);

        if (requires_grad_) {
            result->grad_fn_ = std::make_shared<NegativeOperation>(
                shared_from_this()
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::pow(const std::shared_ptr<Variable>& other) {
        auto result = create(std::pow(value_, other->value_), requires_grad_ || other->requires_grad_);

        if (requires_grad_ || other->requires_grad_) {
            result->grad_fn_ = std::make_shared<PowerOperation>(
                shared_from_this(),
                other
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::log() {
        auto result = create(std::log(value_), requires_grad_);

        if (requires_grad_) {
            result->grad_fn_ = std::make_shared<LogarithmOperation>(
                shared_from_this()
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::exp() {
        auto result = create(std::exp(value_), requires_grad_);

        if (requires_grad_) {
            result->grad_fn_ = std::make_shared<ExponentialOperation>(
                shared_from_this()
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::sin() {
        auto result = create(std::sin(value_), requires_grad_);

        if (requires_grad_) {
            result->grad_fn_ = std::make_shared<SineOperation>(
                shared_from_this()
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::cos() {
        auto result = create(std::cos(value_), requires_grad_);

        if (requires_grad_) {
            result->grad_fn_ = std::make_shared<CosineOperation>(
                shared_from_this()
            );
        }

        return result;
    }

    std::shared_ptr<Variable> Variable::tanh() {
        auto result = create(std::tanh(value_), requires_grad_);

        if (requires_grad_) {
            result->grad_fn_ = std::make_shared<TanhOperation>(
                shared_from_this()
            );
        }

        return result;
    }

    void Variable::print() const {
        std::cout << "Variable(value=" << value_ << ", grad=" << grad_ << ")" << std::endl;
    }

    std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs) {
        return lhs->operator+(rhs);
    }

    std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs) {
        return lhs->operator-(rhs);
    }

    std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& lhs) {
        return lhs->operator-();
    }

    std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs) {
        return lhs->operator*(rhs);
    }

    std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs) {
        return lhs->operator/(rhs);
    }

    std::shared_ptr<Variable> pow(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs) {
        return lhs->pow(rhs);
    }

    std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& lhs, const double rhs) {
        const auto rhs_var = Variable::create(rhs, false);
        return lhs->operator+(rhs_var);
    }

    std::shared_ptr<Variable> operator+(const double lhs, const std::shared_ptr<Variable>& rhs) {
        const auto lhs_var = Variable::create(lhs, false);
        return lhs_var->operator+(rhs);
    }

    std::shared_ptr<Variable> operator*(std::shared_ptr<Variable>& lhs, const double rhs) {
        const auto rhs_var = Variable::create(rhs, false);
        return lhs * rhs_var;
    }

    std::shared_ptr<Variable> operator*(const double lhs, const std::shared_ptr<Variable>& rhs) {
        const auto lhs_var = Variable::create(lhs, false);
        return lhs_var * rhs;
    }

    std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& lhs, const double rhs) {
        const auto rhs_var = Variable::create(rhs, false);
        return lhs - rhs_var;
    }

    std::shared_ptr<Variable> operator-(const double lhs, const std::shared_ptr<Variable>& rhs) {
        const auto lhs_var = Variable::create(lhs, false);
        return lhs_var - rhs;
    }

    std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& lhs, const double rhs) {
        const auto rhs_var = Variable::create(rhs, false);
        return lhs / rhs_var;
    }

    std::shared_ptr<Variable> operator/(const double lhs, const std::shared_ptr<Variable>& rhs) {
        const auto lhs_var = Variable::create(lhs, false);
        return lhs_var / rhs;
    }

    std::shared_ptr<Variable> pow(const std::shared_ptr<Variable>& lhs, const double rhs) {
        const auto rhs_var = Variable::create(rhs, false);
        return pow(lhs, rhs_var);
    }

    std::shared_ptr<Variable> pow(const double lhs, const std::shared_ptr<Variable>& rhs) {
        const auto lhs_var = Variable::create(lhs, false);
        return pow(lhs_var, rhs);
    }

}