#include "autodiff/operation/operation.h"

namespace autodiff {

    class NegativeOperation final : public Operation {
    public:
        explicit NegativeOperation(Variable* input) : in(input) {}

        void backward(const double grad_output) override {
            if (in->requires_grad()) in->grad_ -= grad_output;
        }

        std::vector<Variable*> get_inputs() override {
            return {in};
        }

    private:
        Variable* in;
    };

}
