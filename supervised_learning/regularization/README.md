### L1 Regularization
- **Description:** Uses the absolute value of the magnitude of the coefficients multiplied by a constant alpha (the regularization parameter).
- **Purpose:** Creates sparse models, eliminating some coefficients by setting them to zero.
- **Use Case:** Prevents overfitting.

### L2 Regularization
- **Description:** Uses the squared magnitude of the coefficients multiplied by a constant alpha.
- **Purpose:** Prevents overfitting.
- **Use Case:** The most common type of regularization.

### Elastic Net Regularization
- **Description:** A linear combination of L1 and L2 regularization.
- **Purpose:** Prevents overfitting.
- **Use Case:** Useful when there are multiple features that are correlated with one another.

### Dropout
- **Description:** Randomly sets a fraction rate of input units to 0 at each update during training time.
- **Purpose:** Helps prevent overfitting.
- **Mechanism:** The units that are kept are scaled by \(1 / (1 - \text{rate})\), maintaining their sum unchanged during training and inference.

### Early Stopping
- **Description:** Stops training when a monitored quantity (e.g., validation loss) stops improving.
- **Purpose:** Prevents overfitting.
- **Use Case:** Useful for long training sessions.

### Data Augmentation
- **Description:** Increases the size of the training set by adding transformed copies of existing data or artificially created data.
- **Purpose:** Prevents overfitting.
- **Use Case:** Useful when there is insufficient data to train the model.

### Batch Normalization
- **Description:** Normalizes the activations of the previous layer at each batch.
- **Purpose:** Keeps the mean activation close to 0 and the activation standard deviation close to 1.
- **Use Case:** Prevents overfitting, especially useful during long training sessions.

### Gradient Noise Injection
- **Description:** Adds noise to the gradient during training.
- **Purpose:** Prevents overfitting.
- **Use Case:** Useful for long training sessions.

---

```python
# 0x01-regularization/0-l2_reg_cost.py
def l2_reg_cost(cost):
    # Path: supervised_learning/regularization/0-l2_reg_cost.py
    pass

# 0x01-regularization/0-l2_reg_gradient_descent.py
def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using gradient descent with L2 regularization"""
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        if i == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - (A ** 2))
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + ((lambtha / m) * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
    return weights

# 0x05-regularization/0-weights.py
def l2_reg_cost(cost, lambtha, weights, L, m):
    # Path: supervised_learning/regularization/0-weights.py
    """Calculates the cost of a neural network with L2 regularization using TensorFlow"""
    return cost + tf.losses.get_regularization_losses()

# 0x06-keras/3-l2_reg_create_layer.py
def l2_reg_create_layer(prev, n, activation, lambtha):
    # Path: supervised_learning/regularization/0-weights.py
    """Creates a TensorFlow layer that includes L2 regularization"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=kernel, kernel_regularizer=l2)
    return layer(prev)

def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    for i in range(L):
        A = cache['A' + str(i)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        Z = np.matmul(W, A) + b
        if i == L - 1:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = np.where(D < keep_prob, 1, 0)
        A = np.multiply(A, D)
        A /= keep_prob
        cache['D' + str(i + 1)] = D
        cache['A' + str(i + 1)] = A
    return cache

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Conducts forward propagation using Dropout"""
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        if i == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - (A ** 2))
            dZ = np.multiply(dZ, cache['D' + str(i)])
            dZ /= keep_prob
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
    return weights

# 0x07-dropout_create_layer.py
def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=init)
    dropout = tf.layers.Dropout(keep_prob)
    return dropout(layer(prev))

def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return True, count
    return False, count
