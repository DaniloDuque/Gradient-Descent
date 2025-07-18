#include "autodiff/operation/Operation.h"

namespace autodiff {

    class LogarithmOperation final : public Operation {
    public:
        explicit LogarithmOperation(const std::shared_ptr<Variable>& arg)
            : arg(arg), arg_val(arg->value()) {}

        void backward(const double grad_output) override {
            if (arg->requires_grad()) {
                const double local_grad = 1.0 / arg_val;
                arg->grad_ += grad_output * local_grad;
            }
        }

        std::vector<std::shared_ptr<Variable>> get_inputs() override {
            return {arg};
        }

    private:
        std::shared_ptr<Variable> arg;
        double arg_val;
    };

}