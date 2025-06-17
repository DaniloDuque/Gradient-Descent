#include "autodiff/operation/Operation.h"

namespace autodiff {

    class SubtractOperation final : public Operation {
    public:
        SubtractOperation(Variable* left, Variable* right)
            : lft(left), rght(right) {}

        void backward(const double grad_output) override {
            if (lft->requires_grad()) {
                lft->grad_ += grad_output * 1.0;
            }
            if (rght->requires_grad()) {
                rght->grad_ += grad_output * -1.0;
            }
        }

        std::vector<Variable*> get_inputs() override {
            return {lft, rght};
        }

    private:
        Variable* lft;
        Variable* rght;
    };

}
