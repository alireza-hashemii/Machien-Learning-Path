import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

def J_convex(w):
    return w[0] ** 2 + 2 * w[1] ** 2


def grad_J_convex(w):
    return np.array([2 *w[0], 4 *w[1]])



def gradient_descent(grad: Callable, w_init, learning_rate=0.1, n_steps=100 ):
    w = np.array(w_init)
    path = [w.copy()]
    
    for i in range(n_steps):
        gradient = grad(w)
        w = w - learning_rate * gradient
        path.append(w.copy())

    return np.array(path)


if __name__ == '__main__':
    weights = gradient_descent(grad_J_convex, [2, 3])


    # Create a grid of points to evaluate the function
    w0_vals = np.linspace(-3, 3, 100)
    w1_vals = np.linspace(-3, 3, 100)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    J_vals = W0**2 + 2 * W1**2
    plt.figure(figsize=(8,6))
    contours = plt.contour(W0, W1, J_vals, levels=20, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)

    # Plot gradient descent path
    weights = np.array(weights)
    plt.plot(weights[:,0], weights[:,1], marker='o', color='red', label='Gradient Descent Path')
    plt.scatter(weights[0,0], weights[0,1], color='blue', label='Start')
    plt.scatter(weights[-1,0], weights[-1,1], color='green', label='End')

    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('Gradient Descent Path on Contours of J(w0, w1)')
    plt.legend()
    plt.grid(True)
    plt.show()