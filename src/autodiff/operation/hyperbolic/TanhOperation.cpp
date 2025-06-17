#include "autodiff/operation/Operation.h"

namespace autodiff {
    class TanhOperation final : public Operation {
    public:
        explicit TanhOperation(Variable* input)
            : in(input), in_val(input->value()) {}

        void backward(const double grad_output) override {
            if (in->requires_grad()) {
                const double local_grad = std::tanh(in_val);
                in->grad_ += grad_output * (1 - local_grad * local_grad);
            }
        }

        std::vector<Variable*> get_inputs() override {
            return {in};
        }

    private:
        Variable* in;
        double in_val;
    };

}
