#pragma once
#include "autodiff/variable/Variable.h"
#include "optimizers/GradientDescent.h"

class Vanilla final : public GradientDescent {
    std::vector<std::shared_ptr<autodiff::Variable>> y_pred;
public:
    using Vector = std::vector<double>;
    using Matrix = std::vector<Vector>;
    using Variable = std::shared_ptr<autodiff::Variable>;

    void train(std::vector<Variable>& w,
           const Matrix& X,
           const Vector& y_true,
           LossFunction& loss_fn, 
           const double& learning_rate) override {

        const size_t n_samples = y_true.size();
        y_pred.clear();

        const size_t n_features = w.size() - 1; // Last element is bias
        y_pred.reserve(n_samples);

        for (size_t i = 0; i < n_samples; ++i) {
            auto pred = w[n_features];
            
            for (size_t j = 0; j < n_features; ++j) {
                auto x_ij = autodiff::Variable::create(X[i][j]);
                pred = pred + w[j] * x_ij;
            }

            y_pred.push_back(pred);
        }

        const auto loss = loss_fn.compute(y_pred, y_true);
        loss->backward();

        for (const auto& param : w) {
            param->set_value(param->value() - learning_rate * param->grad());
            param->zero_grad();
        }
    }

};
