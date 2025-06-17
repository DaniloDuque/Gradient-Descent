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

    void Variable::backward() {
        grad_ = 1.0;
        std::vector<Variable*> sorted;
        std::set<Variable*> visited;
        topological_sort(sorted, visited);

        for (const auto& it : std::ranges::reverse_view(sorted)) {
            if (it->grad_fn_) it->grad_fn_->backward(it->grad_);
        }
    }

    void Variable::topological_sort(std::vector<Variable*>& sorted, std::set<Variable*>& visited) const {
        if (visited.contains(const_cast<Variable*>(this))) return;

        visited.insert(const_cast<Variable*>(this));

        if (grad_fn_) {
            for (const auto& input : grad_fn_->get_inputs())
                input->topological_sort(sorted, visited);
        }
        
        sorted.push_back(const_cast<Variable*>(this));
    }

    Variable Variable::operator+(const Variable& other) const {
        auto result = Variable(value_ + other.value_, requires_grad_ || other.requires_grad_);
        
        if (requires_grad_ || other.requires_grad_) {
            result.grad_fn_ = std::make_shared<AddOperation>(
                const_cast<Variable*>(this),
                const_cast<Variable*>(&other)
            );
        }
        
        return result;
    }

    Variable Variable::operator-(const Variable& other) const {
        auto result = Variable(value_ - other.value_, requires_grad_ || other.requires_grad_);

        if (requires_grad_ || other.requires_grad_) {
            result.grad_fn_ = std::make_shared<SubtractOperation>(
                const_cast<Variable*>(this),
                const_cast<Variable*>(&other)
            );
        }
        return result;
    }

    Variable Variable::operator*(const Variable& other) const {
        auto result = Variable(value_ * other.value_, requires_grad_ || other.requires_grad_);
        
        if (requires_grad_ || other.requires_grad_) {
            result.grad_fn_ = std::make_shared<MultiplyOperation>(
                const_cast<Variable*>(this),
                const_cast<Variable*>(&other)
            );
        }
        
        return result;
    }

    Variable Variable::operator/(const Variable& other) const {
        auto result = Variable(value_ / other.value_, requires_grad_ || other.requires_grad_);
        
        if (requires_grad_ || other.requires_grad_) {
            result.grad_fn_ = std::make_shared<DivideOperation>(
                const_cast<Variable*>(this),
                const_cast<Variable*>(&other)
            );
        }
        
        return result;
    }

    Variable Variable::operator-() const {
        auto result = Variable(-value_, requires_grad_);
        
        if (requires_grad_) {
            result.grad_fn_ = std::make_shared<NegativeOperation>(
                const_cast<Variable*>(this)
            );
        }
        
        return result;
    }

    Variable Variable::operator^(const Variable& other) const {
        auto result = Variable(std::pow(value_, other.value_), requires_grad_ || other.requires_grad_);

        if (requires_grad_ || other.requires_grad_) {
            result.grad_fn_ = std::make_shared<PowerOperation>(
                const_cast<Variable*>(this),
                const_cast<Variable*>(&other)
            );
        }

        return result;
    }

    Variable Variable::log() const {
        auto result = Variable(std::log(value_), requires_grad_);

        if (requires_grad_) {
            result.grad_fn_ = std::make_shared<LogarithmOperation>(
                const_cast<Variable*>(this)
            );
        }

        return result;
    }

    Variable Variable::exp() const {
        auto result = Variable(std::exp(value_), requires_grad_);

        if (requires_grad_) {
            result.grad_fn_ = std::make_shared<ExponentialOperation>(
                const_cast<Variable*>(this)
            );
        }

        return result;
    }

    Variable Variable::sin() const {
        auto result = Variable(std::sin(value_), requires_grad_);

        if (requires_grad_) {
            result.grad_fn_ = std::make_shared<SineOperation>(
                const_cast<Variable*>(this)
            );
        }

        return result;
    }

    Variable Variable::cos() const {
        auto result = Variable(std::cos(value_), requires_grad_);

        if (requires_grad_) {
            result.grad_fn_ = std::make_shared<CosineOperation>(
                const_cast<Variable*>(this)
            );
        }

        return result;
    }

    Variable Variable::tanh() const {
        auto result = Variable(std::tanh(value_), requires_grad_);

        if (requires_grad_) {
            result.grad_fn_ = std::make_shared<TanhOperation>(
                const_cast<Variable*>(this)
            );
        }

        return result;
    }

    void Variable::print() const {
        std::cout << "Variable(value=" << value_ << ", grad=" << grad_ << ")" << std::endl;
    }

}