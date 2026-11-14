# One input -> one ReLU hidden unit -> one output.
x = 0.8
target = 1.0

# Initial weights and biases
w1, b1 = 0.5, 0.1
w2, b2 = 0.5, 0.1
learning_rate = 0.1


def forward(x, w1, b1, w2, b2):
    """Return the hidden unit's values and the network's prediction."""
    hidden_pre = x * w1 + b1
    hidden = max(0.0, hidden_pre)  # ReLU
    prediction = hidden * w2 + b2
    return hidden_pre, hidden, prediction


# Forward pass
hidden_pre, hidden, prediction = forward(x, w1, b1, w2, b2)
loss = 0.5 * (prediction - target) ** 2
print(f"Before update: hidden={hidden:.3f}, prediction={prediction:.3f}, loss={loss:.4f}")

# Backpropagation: calculate how each parameter contributed to the error.
error = prediction - target

dw2 = error * hidden
db2 = error

dhidden = error * w2
relu_gradient = 1.0 if hidden_pre > 0 else 0.0
dw1 = dhidden * relu_gradient * x
db1 = dhidden * relu_gradient

# Update weights and biases
w1 -= learning_rate * dw1
b1 -= learning_rate * db1
w2 -= learning_rate * dw2
b2 -= learning_rate * db2

# Forward pass again with the updated parameters
_, hidden, prediction = forward(x, w1, b1, w2, b2)
loss = 0.5 * (prediction - target) ** 2
print(f"After update:  hidden={hidden:.3f}, prediction={prediction:.3f}, loss={loss:.4f}")
print(f"Updated parameters: w1={w1:.4f}, b1={b1:.4f}, w2={w2:.4f}, b2={b2:.4f}")