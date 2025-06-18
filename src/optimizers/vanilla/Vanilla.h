#pragma once
#include "autodiff/variable/Variable.h"
#include "optimizers/GradientDescent.h"

class Vanilla final : public GradientDescent {
public:
    using Vector = std::vector<double>;
    using Matrix = std::vector<Vector>;
    using Variable = std::shared_ptr<autodiff::Variable>;

    explicit Vanilla(const double learning_rate) {
        this->learning_rate = learning_rate;
    }

    void train(std::vector<Variable>& w,
           const Matrix& X,
           const Vector& y_true,
           LossFunction& loss_fn) override {

        const size_t n_samples = y_true.size();
        std::vector<Variable> y_pred;

        for (size_t i = 0; i < n_samples; ++i) {
            auto pred = autodiff::Variable::create(0.0);

            for (size_t j = 0; j < w.size(); ++j) {
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
