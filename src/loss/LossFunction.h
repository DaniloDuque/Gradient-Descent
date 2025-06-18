#pragma once
#include <future>
#include <vector>
#include "autodiff/variable/Variable.h"

class LossFunction {
public:
    using Variable = std::shared_ptr<autodiff::Variable>;
    virtual ~LossFunction() = default;
    virtual Variable compute(std::vector<Variable>& y_pred, const std::vector<double>& y_true) = 0;
};
