# Gradient Descent Python Module

This directory contains Python bindings for the C++ gradient descent library, combining automatic differentiation and optimization functionality in a single module.

## Building the Module

1. Make sure you have the required dependencies:
   ```bash
   python3 -m pip install pybind11 setuptools numpy matplotlib
   ```

2. Build the module:
   ```bash
   cd src/notebooks
   python3 setup.py build_ext --inplace
   ```

This will create `gradientdescent.cpython-XXX-darwin.so` (or similar) in this directory.

## Usage

### Automatic Differentiation

```python
import gradientdescent as gd

# Create variables
x = gd.Variable.create(2.0, True)  # requires_grad=True
y = gd.Variable.create(3.0, True)

# Perform operations
z = x * x + 2 * x * y + y * y

# Compute gradients
z.backward()

print(f"∂z/∂x = {x.grad}")  # Should be 2*x + 2*y = 10
print(f"∂z/∂y = {y.grad}")  # Should be 2*x + 2*y = 10
```

### Gradient Descent Optimization

```python
import gradientdescent as gd

# Create data
X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]  # Features
y = [5.0, 8.0, 11.0]  # Target values

# Initialize weights
w = [gd.Variable.create(0.0, True), gd.Variable.create(0.0, True)]

# Create loss function and optimizer
loss_fn = gd.MSE()
optimizer = gd.Vanilla()

# Train for one step
learning_rate = 0.01
optimizer.train(w, X, y, loss_fn, learning_rate)

print(f"Updated weights: [{w[0].value}, {w[1].value}]")
```

## Available Components

### Automatic Differentiation
- **Variable** - Core class for automatic differentiation
- **Mathematical operations**: `+`, `-`, `*`, `/`, `exp()`, `log()`, `sin()`, `cos()`, `tanh()`, `pow()`

### Optimization
- **Loss Functions**: 
  - `MSE` - Mean Squared Error loss function

- **Optimizers**:
  - `Vanilla` - Standard gradient descent optimizer

## Examples

See `tutorial.ipynb` for comprehensive examples including:
- Automatic differentiation with various operations
- Linear regression using gradient descent
- Loss and weight trajectory visualization

## Troubleshooting

If you get import errors:
1. Make sure you're in the `src/notebooks` directory when running Python
2. Rebuild the module if you've made changes to the C++ code
3. Check that all dependencies are installed