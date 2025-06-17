#include "autodiff/operation/Operation.h"

namespace autodiff {

    class ExponentialOperation final : public Operation {
    public:
        explicit ExponentialOperation(const std::shared_ptr<Variable>& input)
            : in(input), in_val(input->value()) {}

        void backward(const double grad_output) override {
            if (in->requires_grad()) {
                const double local_grad = std::exp(in_val);
                in->grad_ += grad_output * local_grad;
            }
        }

        std::vector<std::shared_ptr<Variable>> get_inputs() override {
            return {in};
        }

    private:
        std::shared_ptr<Variable> in;
        double in_val;
    };

}