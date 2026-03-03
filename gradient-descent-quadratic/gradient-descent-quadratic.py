def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = float(x0)  # x = x0 - lr * f'x
    for _ in range(steps):
        grad = 2*a*x + b
        x = x - lr *grad

    return x