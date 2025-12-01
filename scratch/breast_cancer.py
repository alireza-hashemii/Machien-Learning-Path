import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def main():
    global X_train, X_test, y_train, y_test 
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    # Select two features (mean perimeter and mean smoothness)
    X =  dataset.data[:, [2, 4]]
    
    scaler  =StandardScaler()
    # Standardize features
    X = scaler.fit_transform(X)

    # Add bias term (column of ones)
    X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

    # Split data into training and validation sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, random_state=42, train_size=.8)
    plot(X_train, y_train)


# Sigmoid activation function(to scale the output of wT.X)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# Compute logistic regression loss (binary cross-entropy)
def compute_loss(w, X, y):
    z = X @ w
    predictions = sigmoid(z)
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return loss


# Compute gradient of the loss weights
def compute_gradient(w, X, y):
    z = X @ w
    predictions = sigmoid(z)
    errors = predictions - y
    gradient = X.T @ errors / len(y)
    return gradient


# Evaluate validation accuracy
def validation_accuracy(w, X_val, y_val):
    probabilities = sigmoid(X_val @ w)
    predictions = (probabilities > 0.5).astype(int)
    accuracy = np.mean(predictions == y_val)
    return accuracy



# Gradient descent for logistic regression
def gradient_descent_logistic(X_train, y_train, X_val, y_val, learning_rate=0.1, n_steps=1000, tolerance=1e-6):
    # Initialize weights
    w = np.zeros(X_train.shape[1])
    # Track loss, accuracy, and weights
    loss_history = [compute_loss(w, X_train, y_train)]
    val_accuracy_history = [validation_accuracy(w, X_val, y_val)]
    weights_history = [w.copy()]

    # Main training loop
    for step in range(1, n_steps + 1):
        grad = compute_gradient(w, X_train, y_train)
        w -= learning_rate * grad  # Update weights

        loss = compute_loss(w, X_train, y_train)
        loss_history.append(loss)

        acc = validation_accuracy(w, X_val, y_val)
        val_accuracy_history.append(acc)

        # Save weights every 10 steps
        if step % 10 == 0:
            weights_history.append(w.copy())

        # Check for convergence
        if np.abs(loss_history[-2] - loss_history[-1]) < tolerance:
            print(f'Converged at step {step}')
            break

        # Log progress every 100 steps
        if step % 100 == 0:
            print(f'Step {step}: Loss = {loss:.4f}, Validation Accuracy = {acc:.4f}')

    return w, loss_history, val_accuracy_history, weights_history




def plot(X_train, y_train):
        # Plot training data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                color='red', label='Malignant')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                color='blue', label='Benign')
    plt.xlabel('Mean Perimeter (Standardized)')
    plt.ylabel('Mean Smoothness (Standardized)')
    plt.title('Breast Cancer Dataset (Training Set)')
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()
    # Set learning rate and number of steps
    learning_rate = 0.05
    n_steps = 800

    # Train logistic regression model with gradient descent
    w_opt, loss_history, val_accuracy_history, weights_history = gradient_descent_logistic(
        X_train, y_train, X_test, y_test,
        learning_rate=learning_rate,
        n_steps=n_steps,
    )

    print(f'Optimized weights: {w_opt}')
    print(f'Decision rule: {w_opt[0]} * Mean Perimeter + {w_opt[1]} * Mean Smoothness + {w_opt[2]} > 0 : Benign')
    print(f'Decision rule: {w_opt[0]} * Mean Perimeter + {w_opt[1]} * Mean Smoothness + {w_opt[2]} < 0 : Malignant')