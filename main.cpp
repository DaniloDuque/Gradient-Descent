#include "optimizers/vanilla/Vanilla.h"
#include "loss/mse/MSE.h"
#include <iomanip>
#include <iostream>

using autodiff::Variable;
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

int main() {
    Matrix X({
        {1.0, 0.0, 2.0, 1.0, 3.0},
        {0.0, 1.0, 1.0, 0.0, 2.0},
        {2.0, 3.0, 0.5, 1.5, 4.0},
        {1.0, 2.0, 0.0, 0.0, 1.0},
        {0.5, 1.0, 1.5, 1.0, 0.0},
        {3.0, 2.0, 2.0, 1.0, 1.0},
        {2.0, 1.0, 0.0, 0.5, 0.0},
        {1.0, 1.0, 1.0, 2.0, 2.0},
        {0.0, 2.0, 2.0, 0.0, 1.0},
        {1.0, 3.0, 0.5, 1.5, 0.0}
    });

    // y = 2x₁ + 3x₂ + 0.5x₃ + 1.5x₄ + 4x₅
    Vector y_true({
        2*1 + 3*0 + 0.5*2 + 1.5*1 + 4*3,
        2*0 + 3*1 + 0.5*1 + 1.5*0 + 4*2,
        2*2 + 3*3 + 0.5*0.5 + 1.5*1.5 + 4*4,
        2*1 + 3*2 + 0.5*0 + 1.5*0 + 4*1,
        2*0.5 + 3*1 + 0.5*1.5 + 1.5*1 + 4*0,
        2*3 + 3*2 + 0.5*2 + 1.5*1 + 4*1,
        2*2 + 3*1 + 0.5*0 + 1.5*0.5 + 4*0,
        2*1 + 3*1 + 0.5*1 + 1.5*2 + 4*2,
        2*0 + 3*2 + 0.5*2 + 1.5*0 + 4*1,
        2*1 + 3*3 + 0.5*0.5 + 1.5*1.5 + 4*0
    });

    std::vector w = {
        Variable::create(0.1, true),
        Variable::create(0.1, true),
        Variable::create(0.1, true),
        Variable::create(0.1, true),
        Variable::create(0.1, true)
    };

    Vanilla optimizer;
    MSE loss_fn;
    constexpr double learning_rate = 0.01;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ Gradient Descent Training ================\n";

    for (int epoch = 0; epoch < 100; ++epoch) {
        optimizer.train(w, X, y_true, loss_fn, learning_rate);

        std::vector<std::shared_ptr<Variable>> y_pred;
        for (size_t i = 0; i < y_true.size(); ++i) {
            auto pred = Variable::create(0.0);
            for (size_t j = 0; j < w.size(); ++j) {
                pred = pred + w[j] * Variable::create(X[i][j]);
            }
            y_pred.push_back(pred);
        }

        const auto loss = loss_fn.compute(y_pred, y_true);

        std::cout << "Epoch " << std::setw(3) << epoch + 1 << " | "
                  << "Loss: " << loss->value() << " | "
                  << "Weights: [ ";
        for (const auto& param : w) {
            std::cout << param->value() << " ";
        }
        std::cout << "]\n";
    }

    std::cout << "===========================================================\n";
    std::cout << "Final Weights:\n";
    for (size_t i = 0; i < w.size(); ++i) {
        std::cout << "  w[" << i << "] = " << w[i]->value() << '\n';
    }

    return 0;
}
