#include "Variable.h"
#include <iostream>
#include <cmath>
#include "autodiff/operation/Operation.h"
#include "autodiff/operation/arithmetic/AddOperation.cpp"
#include "autodiff/operation/arithmetic/MultiplyOperation.cpp"
#include "autodiff/operation/arithmetic/DivideOperation.cpp"
#include "autodiff/operation/unary/NegativeOperation.cpp"
#include "autodiff/operation/expolog/PowerOperation.cpp"

namespace autodiff {

    Variable::Variable(double value, bool requires_grad)
        : value_(value), grad_(0.0), requires_grad_(requires_grad), grad_fn_(nullptr) {}

    Variable::Variable(double value, bool requires_grad, std::shared_ptr<Operation> grad_fn)
        : value_(value), grad_(0.0), requires_grad_(requires_grad), grad_fn_(std::move(grad_fn)) {}

    void Variable::backward() {
        grad_ = 1.0;
        std::vector<Variable*> sorted;
        std::set<Variable*> visited;
        topological_sort(sorted, visited);
        
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
            if ((*it)->grad_fn_) (*it)->grad_fn_->backward((*it)->grad_);
        }
    }

    void Variable::topological_sort(std::vector<Variable*>& sorted, std::set<Variable*>& visited) const {
        if (visited.find(const_cast<Variable*>(this)) != visited.end()) {
            return;
        }
        
        visited.insert(const_cast<Variable*>(this));
        
        if (grad_fn_) {
            auto inputs = grad_fn_->get_inputs();
            for (auto& input : inputs) {
                input->topological_sort(sorted, visited);
            }
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
        return *this + (-other);
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

    Variable Variable::log(const Variable& other) const {
        auto result = Variable(std::log(value_) / std::log(other.value_), requires_grad_ || other.requires_grad_);
        if (requires_grad_ || other.requires_grad_) {
            result.grad_fn_ = std::make_shared<LogarithmOperation>(
                const_cast<Variable*>(this),
                const_cast<Variable*>(&other)
            );
        }
    }

    Variable Variable::exp() const {
        // TODO: Implement exp operation
        return Variable(std::exp(value_), false);
    }

    Variable Variable::log() const {
        // TODO: Implement log operation
        return Variable(std::log(value_), false);
    }

    Variable Variable::sin() const {
        // TODO: Implement sin operation
        return Variable(std::sin(value_), false);
    }

    Variable Variable::cos() const {
        // TODO: Implement cos operation
        return Variable(std::cos(value_), false);
    }

    Variable Variable::tanh() const {
        // TODO: Implement tanh operation
        double tanh_val = std::tanh(value_);
        return Variable(tanh_val, false);
    }

    Variable Variable::relu() const {
        // TODO: Implement relu operation
        return Variable(value_ > 0 ? value_ : 0, false);
    }

    Variable Variable::sigmoid() const {
        // TODO: Implement sigmoid operation
        double sig_val = 1.0 / (1.0 + std::exp(-value_));
        return Variable(sig_val, false);
    }

    void Variable::print() const {
        std::cout << "Variable(value=" << value_ << ", grad=" << grad_ << ")" << std::endl;
    }

}