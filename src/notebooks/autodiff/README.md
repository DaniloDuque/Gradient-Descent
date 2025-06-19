# Autodiff Python Module

This directory contains Python bindings for the C++ autodiff library, allowing you to use automatic differentiation from Jupyter notebooks and Python scripts.

## Building the Module

1. Make sure you have the required dependencies:
   ```bash
   python3 -m pip install pybind11 setuptools
   ```

2. Build the module:
   ```bash
   cd src/notebooks/autodiff
   python3 setup.py build_ext --inplace
   ```

This will create `autodiff.cpython-313-darwin.so` (or similar) in this directory.

## Usage

### Basic Example

```python
import autodiff

# Create variables
x = autodiff.Variable.create(2.0, True)  # value=2.0, requires_grad=True
y = autodiff.Variable.create(3.0, True)

# Perform operations
z = x * x + 2 * x * y + y * y

# Compute gradients
z.backward()

print(f"∂z/∂x = {x.grad}")  # Should be 2*x + 2*y = 10
print(f"∂z/∂y = {y.grad}")  # Should be 2*x + 2*y = 10
```

### Available Operations

- **Arithmetic**: `+`, `-`, `*`, `/`, unary `-`
- **Mathematical functions**: `exp()`, `log()`, `sin()`, `cos()`, `tanh()`, `pow()`
- **Mixed operations**: Variables can be combined with Python floats

### Variable Methods

- `Variable.create(value, requires_grad=False)` - Create a new variable
- `value` - Get the current value
- `grad` - Get the current gradient
- `backward()` - Compute gradients via backpropagation
- `zero_grad()` - Reset gradient to zero
- `set_value(new_value)` - Update the variable's value

## Examples

See `autodiff.ipynb` for comprehensive examples including:
- Basic arithmetic operations
- Gradient computation
- Mathematical functions
- Simple optimization with gradient descent

## Troubleshooting

If you get import errors:
1. Make sure you're in the `src/notebooks/autodiff` directory when running Python
2. Rebuild the module if you've made changes to the C++ code
3. Check that all dependencies are installed