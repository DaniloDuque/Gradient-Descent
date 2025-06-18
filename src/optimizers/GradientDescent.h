#pragma once
#include "autodiff/variable/Variable.h"
#include "loss/LossFunction.h"

class GradientDescent {
public:
    using Vector = std::vector<double>;
    using Matrix = std::vector<Vector>;
    using Variable = std::shared_ptr<autodiff::Variable>;

    virtual ~GradientDescent() = default;

    virtual void train(std::vector<Variable>& w,
                       const Matrix& X,
                       const Vector& y_true,
                       LossFunction& loss_fn, 
                       const double& learning_rate) = 0;
                       
};
