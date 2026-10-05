"""
Check backpropagation by hand on a tiny 1-1-1 network.

    x  --(w2, b2)-->  neuron 2  --(w3, b3)-->  neuron 3  -->  cost

Part 1 computes the gradients with the four backprop equations (BP1-BP4).
Part 2 estimates the same gradients by brute force: nudge each parameter
       a tiny bit up and down and see how much the cost changes.
Part 3 uses the backprop gradients to run gradient descent and shows
       the cost going down.
"""

import math


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


# Training example and starting parameters (same numbers as the worked example)
x = 1.0   # input, a1
y = 1.0   # correct answer
params = {"w2": 0.5, "b2": 0.0, "w3": 2.0, "b3": -1.0}


def forward(p):
    """Forward pass: compute every z and a, left to right."""
    a1 = x
    z2 = p["w2"] * a1 + p["b2"]
    a2 = sigmoid(z2)
    z3 = p["w3"] * a2 + p["b3"]
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3


def cost(p):
    a3 = forward(p)[-1]
    return 0.5 * (y - a3) ** 2


def backprop(p):
    """Backward pass: the four fundamental equations."""
    a1, z2, a2, z3, a3 = forward(p)

    delta3 = (a3 - y) * sigmoid_prime(z3)          # BP1: error at the output
    delta2 = delta3 * p["w3"] * sigmoid_prime(z2)  # BP2: pass the error back

    return {
        "w2": a1 * delta2,   # BP4: incoming value x receiving neuron's error
        "b2": delta2,        # BP3: bias gradient = error
        "w3": a2 * delta3,   # BP4
        "b3": delta3,        # BP3
    }


def numerical_gradient(p, eps=1e-6):
    """Brute force: nudge each parameter up and down, measure the cost change."""
    grads = {}
    for name in p:
        up = dict(p)
        down = dict(p)
        up[name] += eps
        down[name] -= eps
        grads[name] = (cost(up) - cost(down)) / (2 * eps)
    return grads


# ---------------- Part 1 and 2: compare the two methods ----------------
a1, z2, a2, z3, a3 = forward(params)
print("Forward pass")
print(f"  z2 = {z2:.4f}   a2 = {a2:.4f}")
print(f"  z3 = {z3:.4f}   a3 = {a3:.4f}   (correct answer y = {y})")
print(f"  cost = {cost(params):.4f}")
print()

bp = backprop(params)
num = numerical_gradient(params)

print("Gradients: backprop vs brute force")
print(f"  {'param':<6}{'backprop':>12}{'brute force':>14}{'difference':>14}")
for name in params:
    diff = abs(bp[name] - num[name])
    print(f"  {name:<6}{bp[name]:>12.6f}{num[name]:>14.6f}{diff:>14.2e}")
print()
print("Differences around 1e-10 or smaller mean the backprop equations are right.")
print()

# ---------------- Part 3: use the gradients to learn ----------------
eta = 3.0
p = dict(params)
print(f"Gradient descent with eta = {eta}")
for step in range(51):
    if step % 10 == 0:
        print(f"  step {step:>3}: output a3 = {forward(p)[-1]:.4f}   cost = {cost(p):.6f}")
    g = backprop(p)
    for name in p:
        p[name] -= eta * g[name]   # the update rule from Chapter 1
