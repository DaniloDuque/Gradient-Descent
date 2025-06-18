#include "src/autodiff/variable/Variable.h"

using namespace autodiff;

int main() {
    const auto x = Variable::create(2.0, true);
    const auto y = 1.0 / (1.0 + (-x)->exp());

    y->backward();

    x->print();
    y->print();

    return 0;
}
