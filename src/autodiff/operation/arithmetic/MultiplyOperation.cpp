#include "autodiff/operation/Operation.h"

namespace autodiff {

	class MultiplyOperation final : public Operation {
	public:
    	MultiplyOperation(const std::shared_ptr<Variable>& left, const std::shared_ptr<Variable>& right)
        	: lft(left), rght(right), lft_val(left->value()), rght_val(right->value()) {}

    	void backward(const double grad_output) override {
        	if (lft->requires_grad()) {
        		const double local_left_grad = rght_val;
        		lft->grad_ += grad_output * local_left_grad;
        	}
        	if (rght->requires_grad()) {
        		const double local_right_grad = lft_val;
        		rght->grad_ += grad_output * local_right_grad;
        	}
    	}

    	std::vector<std::shared_ptr<Variable>> get_inputs() override {
        	return {lft, rght};
    	}

	private:
    	std::shared_ptr<Variable> lft{};
    	std::shared_ptr<Variable> rght{};
    	double lft_val;
    	double rght_val;
	};

}