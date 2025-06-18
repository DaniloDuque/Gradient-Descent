#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "autodiff/variable/Variable.h"
#include "optimizers/GradientDescent.h"
#include "optimizers/vanilla/Vanilla.h"
#include "loss/LossFunction.h"
#include "loss/mse/MSE.h"

namespace py = pybind11;

PYBIND11_MODULE(gradient_descent, m) {
    m.doc() = "Python bindings for C++ Gradient Descent implementation";
    
    // Bind the Variable class
    py::class_<autodiff::Variable, std::shared_ptr<autodiff::Variable>>(m, "Variable")
        .def_static("create", py::overload_cast<double, bool>(&autodiff::Variable::create),
                   py::arg("value"), py::arg("requires_grad") = false)
        .def("value", &autodiff::Variable::value)
        .def("grad", &autodiff::Variable::grad)
        .def("requires_grad", &autodiff::Variable::requires_grad)
        .def("set_value", &autodiff::Variable::set_value)
        .def("set_grad", &autodiff::Variable::set_grad)
        .def("zero_grad", &autodiff::Variable::zero_grad)
        .def("add_grad", &autodiff::Variable::add_grad)
        .def("backward", &autodiff::Variable::backward)
        .def("print", &autodiff::Variable::print)
        // Math functions
        .def("exp", &autodiff::Variable::exp)
        .def("log", &autodiff::Variable::log)
        .def("pow", &autodiff::Variable::pow)
        .def("tanh", &autodiff::Variable::tanh)
        .def("cos", &autodiff::Variable::cos)
        .def("sin", &autodiff::Variable::sin);
    
    // Add global operators for Variable
    m.def("add", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a + b; });
    m.def("subtract", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a - b; });
    m.def("multiply", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a * b; });
    m.def("divide", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a / b; });
    m.def("negate", [](const std::shared_ptr<autodiff::Variable>& a) { return -a; });
    m.def("pow", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return autodiff::pow(a, b); });
    
    // Variable-scalar operations
    m.def("add", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return a + b; });
    m.def("add", [](double a, const std::shared_ptr<autodiff::Variable>& b) { return a + b; });
    m.def("subtract", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return a - b; });
    m.def("subtract", [](double a, const std::shared_ptr<autodiff::Variable>& b) { return a - b; });
    m.def("multiply", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return a * b; });
    m.def("multiply", [](double a, const std::shared_ptr<autodiff::Variable>& b) { return a * b; });
    m.def("divide", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return a / b; });
    m.def("divide", [](double a, const std::shared_ptr<autodiff::Variable>& b) { return a / b; });
    m.def("pow", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return autodiff::pow(a, b); });
    m.def("pow", [](double a, const std::shared_ptr<autodiff::Variable>& b) { return autodiff::pow(a, b); });
    
    // Python special methods for operators
    py::class_<autodiff::Variable, std::shared_ptr<autodiff::Variable>>(m, "VariableOps")
        .def("__add__", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a + b; })
        .def("__sub__", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a - b; })
        .def("__mul__", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a * b; })
        .def("__truediv__", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return a / b; })
        .def("__neg__", [](const std::shared_ptr<autodiff::Variable>& a) { return -a; })
        .def("__pow__", [](const std::shared_ptr<autodiff::Variable>& a, const std::shared_ptr<autodiff::Variable>& b) { return autodiff::pow(a, b); })
        .def("__radd__", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return b + a; })
        .def("__rsub__", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return b - a; })
        .def("__rmul__", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return b * a; })
        .def("__rtruediv__", [](const std::shared_ptr<autodiff::Variable>& a, double b) { return b / a; });
    
    // Bind the LossFunction abstract class
    py::class_<LossFunction, std::shared_ptr<LossFunction>>(m, "LossFunction")
        .def("compute", &LossFunction::compute);
    
    // Bind the MSE class
    py::class_<MSE, LossFunction, std::shared_ptr<MSE>>(m, "MSE")
        .def(py::init<>())
        .def("compute", &MSE::compute);
    
    // Bind the GradientDescent abstract class
    py::class_<GradientDescent, std::shared_ptr<GradientDescent>>(m, "GradientDescent")
        .def("train", &GradientDescent::train);
    
    // Bind the Vanilla optimizer
    py::class_<Vanilla, GradientDescent, std::shared_ptr<Vanilla>>(m, "Vanilla")
        .def(py::init<>())
        .def("train", &Vanilla::train);
}
