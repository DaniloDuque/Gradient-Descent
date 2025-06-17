#include "autodiff/operation/Operation.h"

namespace autodiff {

    class NegativeOperation final : public Operation {
    public:
        explicit NegativeOperation(const std::shared_ptr<Variable> &input) : in(input) {}

        void backward(const double grad_output) override {
            if (in->requires_grad()) {
                constexpr double local_grad = -1.0;
                in->grad_ += grad_output * local_grad;
            }
        }

        std::vector<std::shared_ptr<Variable>> get_inputs() override {
            return {in};
        }

    private:
        std::shared_ptr<Variable> in;
    };

}
