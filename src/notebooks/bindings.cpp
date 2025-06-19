#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

// AutoDiff includes
#include "autodiff/variable/Variable.h"

// Optimizer includes
#include "loss/LossFunction.h"
#include "loss/mse/MSE.h"
#include "optimizers/GradientDescent.h"
#include "optimizers/vanilla/Vanilla.h"

namespace py = pybind11;

PYBIND11_MODULE(gradientdescent, m) {
    m.doc() = "Gradient descent optimization and automatic differentiation module";

    // ======== AutoDiff Bindings ========
    py::class_<autodiff::Variable, std::shared_ptr<autodiff::Variable>>(m, "Variable")
        .def_static("create",
            [](const double value, const bool requires_grad = false) {
                return autodiff::Variable::create(value, requires_grad);
            },
            "Create a Variable with value and optional gradient requirement",
            py::arg("value"), py::arg("requires_grad") = false)
        
        .def_property_readonly("value", &autodiff::Variable::value,
            "Get the current value of the variable")
        .def_property_readonly("grad", &autodiff::Variable::grad,
            "Get the current gradient of the variable")
        .def_property_readonly("requires_grad", &autodiff::Variable::requires_grad,
            "Check if the variable requires gradient computation")
        
        .def("set_value", &autodiff::Variable::set_value,
            "Set the value of the variable",
            py::arg("new_value"))
        .def("set_grad", &autodiff::Variable::set_grad,
            "Set the gradient of the variable",
            py::arg("grad"))
        .def("zero_grad", &autodiff::Variable::zero_grad,
            "Reset the gradient to zero")
        .def("add_grad", &autodiff::Variable::add_grad,
            "Add to the current gradient",
            py::arg("grad"))
        
        .def("backward", &autodiff::Variable::backward,
            "Compute gradients via backpropagation")
        
        .def("__add__",
            [](const std::shared_ptr<autodiff::Variable>& self, const std::shared_ptr<autodiff::Variable>& other) {
                return self->operator+(other);
            })
        .def("__sub__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const std::shared_ptr<autodiff::Variable>& other) {
                return self->operator-(other);
            })
        .def("__mul__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const std::shared_ptr<autodiff::Variable>& other) {
                return self->operator*(other);
            })
        .def("__truediv__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const std::shared_ptr<autodiff::Variable>& other) {
                return self->operator/(other);
            })
        .def("__neg__", 
            [](const std::shared_ptr<autodiff::Variable>& self) {
                return self->operator-();
            })
        
        .def("__add__",
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator+(self, other);
            })
        .def("__radd__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator+(other, self);
            })
        .def("__sub__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator-(self, other);
            })
        .def("__rsub__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator-(other, self);
            })
        .def("__mul__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator*(self, other);
            })
        .def("__rmul__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator*(other, self);
            })
        .def("__truediv__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator/(self, other);
            })
        .def("__rtruediv__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::operator/(other, self);
            })
        
        .def("exp", &autodiff::Variable::exp,
            "Compute exponential function")
        .def("log", &autodiff::Variable::log,
            "Compute natural logarithm")
        .def("pow", &autodiff::Variable::pow,
            "Compute power function",
            py::arg("exponent"))
        .def("tanh", &autodiff::Variable::tanh,
            "Compute hyperbolic tangent")
        .def("sin", &autodiff::Variable::sin,
            "Compute sine function")
        .def("cos", &autodiff::Variable::cos,
            "Compute cosine function")
        
        .def("__pow__",
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::pow(self, other);
            })
        .def("__rpow__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const double other) {
                return autodiff::pow(other, self);
            })
        .def("__pow__", 
            [](const std::shared_ptr<autodiff::Variable>& self, const std::shared_ptr<autodiff::Variable> &other) {
                return autodiff::pow(self, other);
            })
        
        .def("print", &autodiff::Variable::print,
            "Print the variable's value and gradient")
        
        .def("__repr__",
            [](const autodiff::Variable& v) {
                return "Variable(value=" + std::to_string(v.value()) + 
                       ", grad=" + std::to_string(v.grad()) + 
                       ", requires_grad=" + (v.requires_grad() ? "True" : "False") + ")";
            })
        .def("__str__", 
            [](const autodiff::Variable& v) {
                return "Variable(" + std::to_string(v.value()) + ")";
            })
        .def("__format__",
            [](const autodiff::Variable& v, const std::string& format_spec) {
                return std::to_string(v.value());
            });

    // AutoDiff math functions
    m.def("exp", [](const std::shared_ptr<autodiff::Variable>& x) { return x->exp(); },
        "Compute exponential function", py::arg("x"));
    
    m.def("log", [](const std::shared_ptr<autodiff::Variable>& x) { return x->log(); },
        "Compute natural logarithm", py::arg("x"));
    
    m.def("sin", [](const std::shared_ptr<autodiff::Variable>& x) { return x->sin(); },
        "Compute sine function", py::arg("x"));
    
    m.def("cos", [](const std::shared_ptr<autodiff::Variable>& x) { return x->cos(); },
        "Compute cosine function", py::arg("x"));
    
    m.def("tanh", [](const std::shared_ptr<autodiff::Variable>& x) { return x->tanh(); },
        "Compute hyperbolic tangent", py::arg("x"));
    
    m.def("pow", [](const std::shared_ptr<autodiff::Variable>& base, const double exponent) {
        return autodiff::pow(base, exponent); 
    }, "Compute power function", py::arg("base"), py::arg("exponent"));
    
    m.def("pow", [](const double base, const std::shared_ptr<autodiff::Variable>& exponent) {
        return autodiff::pow(base, exponent); 
    }, "Compute power function", py::arg("base"), py::arg("exponent"));

    // ======== Optimizer Bindings ========
    
    // Bind LossFunction base class (abstract)
    py::class_<LossFunction, std::shared_ptr<LossFunction>>(m, "LossFunction")
        .def("compute", &LossFunction::compute, "Compute the loss value",
             py::arg("y_pred"), py::arg("y_true"));

    // Bind MSE loss function
    py::class_<MSE, LossFunction, std::shared_ptr<MSE>>(m, "MSE")
        .def(py::init<>())
        .def("compute", &MSE::compute, "Compute the mean squared error loss",
             py::arg("y_pred"), py::arg("y_true"));

    // Bind GradientDescent base class (abstract)
    py::class_<GradientDescent, std::shared_ptr<GradientDescent>>(m, "GradientDescent")
        .def("train", &GradientDescent::train, "Train the model using gradient descent",
             py::arg("w"), py::arg("X"), py::arg("y_true"), py::arg("loss_fn"), py::arg("learning_rate"));

    // Bind Vanilla gradient descent
    py::class_<Vanilla, GradientDescent, std::shared_ptr<Vanilla>>(m, "Vanilla")
        .def(py::init<>())
        .def("train", &Vanilla::train, "Train the model using vanilla gradient descent",
             py::arg("w"), py::arg("X"), py::arg("y_true"), py::arg("loss_fn"), py::arg("learning_rate"));
}