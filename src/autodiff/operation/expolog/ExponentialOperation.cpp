#include "autodiff/operation/Operation.h"

namespace autodiff {

    class ExponentialOperation final : public Operation {
    public:
        explicit ExponentialOperation(Variable* input)
            : in(input), in_val(input->value()) {}

        void backward(const double grad_output) override {
            if (in->requires_grad()) {
                const double local_grad = std::exp(in_val);
                in->grad_ += grad_output * local_grad;
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