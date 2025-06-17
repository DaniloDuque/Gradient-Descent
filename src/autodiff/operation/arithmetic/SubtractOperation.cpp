#include "autodiff/operation/Operation.h"

namespace autodiff {

    class SubtractOperation final : public Operation {
    public:
        SubtractOperation(const std::shared_ptr<Variable> &left, const std::shared_ptr<Variable> &right)
            : lft(left), rght(right) {}

        void backward(const double grad_output) override {
            if (lft->requires_grad()) {
                constexpr double local_left_grad = 1.0;
                lft->grad_ += grad_output * local_left_grad;
            }
            if (rght->requires_grad()) {
                constexpr double local_right_grad = -1.0;
                rght->grad_ += grad_output * local_right_grad;
            }
        }

        std::vector<std::shared_ptr<Variable>> get_inputs() override {
            return {lft, rght};
        }

    private:
        std::shared_ptr<Variable> lft;
        std::shared_ptr<Variable> rght;
    };

}
