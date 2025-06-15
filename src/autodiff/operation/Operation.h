#pragma once
#include "autodiff/variable/Variable.h"
#include <vector>

namespace autodiff {

    class Operation {
    public:
        virtual ~Operation() = default;
        virtual void backward(double grad_output) = 0;
        virtual std::vector<Variable*> get_inputs() = 0;
    };

}
