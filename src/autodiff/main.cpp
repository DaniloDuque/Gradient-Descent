#include "./variable/Variable.h"

using namespace autodiff;

int main() {
    Variable x(2.0, true);
    Variable t(3.0);
    Variable a = x * x;
    Variable b = a + x;
    Variable y = b + t;
    y.backward();

                       // 3. Backpropagate to compute dy/dx

    x.print(); // should show grad = f'(2.0) = 2*2 + 3 = 7
    y.print(); // optional, value = f(2) = 4 + 6 = 10

    return 0;
}
