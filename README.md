# Gradient Descent with Automatic Differentiation

A from-scratch implementation of gradient descent optimization with reverse-mode automatic differentiation, built in C++ with Python bindings.

## What is Gradient Descent?

Gradient descent is a fundamental optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent. Think of it like rolling a ball down a hill - the ball naturally follows the path that decreases elevation most quickly.

### The Core Algorithm

Given a function $f(\theta)$ that we want to minimize, gradient descent updates parameters according to:

$$\theta_{k+1} = \theta_k - \alpha \nabla f(\theta_k)$$

where:
- $\theta_k$ are the current parameters
- $\alpha$ is the **learning rate** (how big steps to take)
- $\nabla f(\theta_k)$ is the **gradient** (direction of steepest increase)

### Why Does This Work?

The gradient $\nabla f(\theta)$ points in the direction where the function increases most rapidly. By moving in the **opposite direction** (hence the minus sign), we move toward lower values of the function.

**Key Insight:** The gradient tells us both the direction and magnitude of the steepest ascent. Moving opposite to this direction leads us toward a minimum.

### Learning Rate: The Critical Parameter

The learning rate $\alpha$ controls how large steps we take:

- **Too small:** Convergence is very slow - like taking tiny steps down a mountain
- **Too large:** We might overshoot the minimum and oscillate wildly
- **Just right:** Fast, stable convergence to the minimum

**Mathematical Condition:** For smooth functions, we need $\alpha \leq \frac{2}{L}$ where $L$ is the Lipschitz constant of the gradient.

### Convergence Behavior

For well-behaved functions (convex and smooth), gradient descent has **linear convergence**:
$$f(θ_k) - f(θ*) ≤ ρ^k [f(θ_0) - f(θ*)]$$

where $\rho < 1$ is the convergence rate and $\theta^*$ is the optimal solution.

**What this means:** The error decreases exponentially fast - each iteration reduces the error by a constant factor.

## Automatic Differentiation: Computing Gradients Efficiently

The key challenge in gradient descent is computing $\nabla f(\theta)$ efficiently. Our implementation uses **reverse-mode automatic differentiation** (backpropagation).

### The Problem with Manual Derivatives

For complex functions, computing derivatives by hand is:
- **Error-prone:** Easy to make algebraic mistakes
- **Time-consuming:** Requires careful application of chain rule
- **Inflexible:** Must recompute for every function change

### How Automatic Differentiation Works

Instead of computing derivatives symbolically, we build a **computational graph** that tracks operations:

1. **Forward Pass:** Compute function values while recording operations
2. **Backward Pass:** Apply chain rule automatically to compute gradients

**Example:** For $f(x,y) = (x + y) \times \sin(x)$:
```
Forward:  z₁ = x + y,  z₂ = sin(x),  f = z₁ × z₂
Backward: ∂f/∂z₁ = z₂,  ∂f/∂z₂ = z₁,  ∂f/∂x = ∂f/∂z₁ + ∂f/∂z₂ × cos(x)
```

### Chain Rule Implementation

For composite functions $h(x) = f(g(x))$, the chain rule gives:
$$\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

Our system automatically applies this through operation classes that know their own derivatives.

### Supported Operations

| Operation | Function | Derivative Rule |
|-----------|----------|-----------------|
| Addition | $f(x,y) = x + y$ | $\frac{\partial f}{\partial x} = 1, \frac{\partial f}{\partial y} = 1$ |
| Multiplication | $f(x,y) = xy$ | $\frac{\partial f}{\partial x} = y, \frac{\partial f}{\partial y} = x$ |
| Division | $f(x,y) = \frac{x}{y}$ | $\frac{\partial f}{\partial x} = \frac{1}{y}, \frac{\partial f}{\partial y} = -\frac{x}{y^2}$ |
| Exponential | $f(x) = e^x$ | $\frac{\partial f}{\partial x} = e^x$ |
| Logarithm | $f(x) = \ln(x)$ | $\frac{\partial f}{\partial x} = \frac{1}{x}$ |
| Power | $f(x,y) = x^y$ | $\frac{\partial f}{\partial x} = yx^{y-1}, \frac{\partial f}{\partial y} = x^y \ln(x)$ |
| Trigonometric | $\sin(x), \cos(x)$ | $\cos(x), -\sin(x)$ |
| Hyperbolic | $\tanh(x)$ | $1 - \tanh^2(x)$ |

## Linear Regression: A Concrete Example

Let's see gradient descent in action with linear regression, where we want to find the best line through data points.

### The Model

We model the relationship: $y = \mathbf{x}^T\boldsymbol{\theta} + \epsilon$

where:
- $\mathbf{x}$ are input features
- $\boldsymbol{\theta}$ are parameters we want to learn
- $\epsilon$ is noise

### The Loss Function

We use **Mean Squared Error (MSE)**:
$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{x}_i^T\boldsymbol{\theta})^2$$

**Intuition:** This measures how far our predictions are from the true values, on average.

### The Training Process

1. **Initialize:** Start with random weights $\boldsymbol{\theta}_0$
2. **Predict:** Compute $\hat{y}_i = \mathbf{x}_i^T\boldsymbol{\theta}_k$ for all samples
3. **Compute Loss:** Calculate MSE between predictions and true values
4. **Compute Gradients:** Use automatic differentiation
5. **Update:** $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha \nabla \mathcal{L}(\boldsymbol{\theta}_k)$
6. **Repeat:** Until convergence

## Why This Implementation?

### Performance Benefits

- **C++ Core:** Fast numerical computations
- **Memory Efficient:** Minimal overhead per operation
- **Scalable:** Linear complexity in problem size

### Flexibility Benefits

- **Modular Design:** Easy to add new operations and optimizers
- **Python Integration:** Familiar interface for data scientists
- **Extensible:** Support for complex model architectures

### Educational Benefits

- **From Scratch:** Understand every component
- **Mathematical Rigor:** Proper implementation of algorithms
- **Practical Applications:** Real-world examples with datasets
  
## Implementation Architecture

### Core Components

1. **Variable Class**
   - Stores values and gradients
   - Tracks computational graph
   - Implements backpropagation

2. **Operation Classes**
   - Define forward computation
   - Define backward (gradient) computation
   - Chain together to form complex functions

3. **Loss Functions**
   - Measure prediction quality
   - Provide gradients for optimization

4. **Optimizers**
   - Update parameters using gradients
   - Implement different update rules

### Python Bindings

The C++ implementation is exposed to Python through pybind11, providing:
- NumPy array integration
- Jupyter notebook compatibility
- Familiar Python syntax

## Applications Demonstrated

### Stock Price Prediction
- **Problem:** Predict next day's stock price from previous 5 days
- **Model:** Linear regression with time series features
- **Dataset:** Amazon stock prices (1997-2021)

### Student Performance Prediction
- **Problem:** Predict student grades from academic and economic factors
- **Model:** Multi-variate linear regression
- **Features:** Enrollment data, economic indicators, past performance

## Performance Characteristics

### Computational Complexity
- **Forward Pass:** $O(mn)$ where $m$ = samples, $n$ = features
- **Backward Pass:** $O(mn)$ for gradient computation
- **Memory Usage:** $O(n)$ for parameters and gradients
