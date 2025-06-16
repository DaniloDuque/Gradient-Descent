#include "autodiff/operation/Operation.h"
#include <cmath>

namespace autodiff {

    class PowerOperation final : public Operation {
    public:
        PowerOperation(Variable* lft, Variable *rght)
            : lft(lft), rght(rght), lft_val(lft->value()), rght_val(rght->value()) {}

        void backward(const double grad_output) override {
            if (lft->requires_grad()) lft->grad_ += grad_output * rght_val * std::pow(lft_val, rght_val - 1);
            if (rght->requires_grad()) rght->grad_ += grad_output * std::log(lft_val) * std::pow(lft_val, rght_val);
        }

        std::vector<Variable*> get_inputs() override {
            return {lft, rght};
        }

    private:
        Variable* lft;
        Variable* rght;
        double lft_val;
        double rght_val;
    };

}
