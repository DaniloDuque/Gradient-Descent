#include "autodiff/operation/Operation.h"

namespace autodiff {

    class DivideOperation final : public Operation {
    public:
        DivideOperation(Variable* left, Variable* right)
            : lft(left), rght(right),
              lft_val(left->value()), rght_val(right->value()) {}

        void backward(const double grad_output) override {
            if (lft->requires_grad()) lft->grad_ += grad_output * (1.0 / rght_val);
            if (rght->requires_grad()) rght->grad_ += grad_output * (-lft_val / (rght_val * rght_val));
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
