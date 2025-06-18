#pragma once
#include "loss/LossFunction.h"

class MSE final : public LossFunction {
    using Variable = std::shared_ptr<autodiff::Variable>;
public:
    Variable compute(std::vector<Variable>& y_pred, const std::vector<double>& y_true) override {
        auto loss = autodiff::Variable::create(0.0, true);

        for (size_t i = 0; i < y_pred.size(); ++i) {
            auto y = autodiff::Variable::create(y_true[i]);
            auto diff = y_pred[i] - y;
            loss = loss + diff * diff;
        }

        return loss / static_cast<double>(y_pred.size());
    }
};
