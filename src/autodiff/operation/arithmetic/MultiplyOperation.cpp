#include "autodiff/operation/Operation.h"

namespace autodiff {

	class MultiplyOperation final : public Operation {
	public:
    	MultiplyOperation(Variable* left, Variable* right)
        	: lft(left), rght(right), lft_val(left->value()), rght_val(right->value()) {}

    	void backward(const double grad_output) override {
        	if (lft->requires_grad()) lft->grad_ += grad_output * rght_val;
        	if (rght->requires_grad()) rght->grad_ += grad_output * lft_val;
    	}

    	std::vector<Variable*> get_inputs() override {
        	return {lft, rght};
    	}

	private:
    	Variable* lft{};
    	Variable* rght{};
    	double lft_val;
    	double rght_val;
	};

}