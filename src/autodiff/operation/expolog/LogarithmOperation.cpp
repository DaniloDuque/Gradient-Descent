#include "autodiff/operation/Operation.h"

namespace autodiff {

    class LogarithmOperation final : public Operation {
    public:
        LogarithmOperation(Variable* base, Variable *arg)
            : base(base), arg(arg), base_val(base->value()), arg_val(arg->value()) {}

        void backward(const double grad_output) override {
            if (base->requires_grad()) base->grad_ += grad_output * -log(arg_val) / (base_val * pow(log(base_val), 2));
            if (arg->requires_grad()) arg->grad_ += grad_output * 1.0 / (arg_val * log(base_val));
        }

        std::vector<Variable*> get_inputs() override {
            return {base, arg};
        }

    private:
        Variable* base;
        Variable* arg;
        double base_val;
        double arg_val;
    };

}