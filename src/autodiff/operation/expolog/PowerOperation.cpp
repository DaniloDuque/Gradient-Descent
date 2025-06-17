#include "autodiff/operation/Operation.h"
#include <cmath>

namespace autodiff {

    class PowerOperation final : public Operation {
    public:
        PowerOperation(const std::shared_ptr<Variable>& lft, const std::shared_ptr<Variable>& rght)
            : lft(lft), rght(rght), lft_val(lft->value()), rght_val(rght->value()) {}

        void backward(const double grad_output) override {
            if (lft->requires_grad()) {
                const double local_left_grad = rght_val * std::pow(lft_val, rght_val - 1);
                lft->grad_ += grad_output * local_left_grad;
            }
            if (rght->requires_grad()) {
                const double local_right_grad = std::log(lft_val) * std::pow(lft_val, rght_val);
                rght->grad_ += grad_output * local_right_grad;
            }
        }

        std::vector<std::shared_ptr<Variable>> get_inputs() override {
            return {lft, rght};
        }

    private:
        std::shared_ptr<Variable> lft;
        std::shared_ptr<Variable> rght;
        double lft_val;
        double rght_val;
    };

}
