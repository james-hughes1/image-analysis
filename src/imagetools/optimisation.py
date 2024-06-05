import matplotlib.pyplot as plt


def gradient_descent(obj, grad, x0, obj_min, eps, lr, max_iters):
    x = x0
    iteration = 0
    x_1_values = [x0[0]]
    x_2_values = [x0[1]]
    obj_values = [obj(x0)]
    while obj(x) - obj_min > eps and iteration < max_iters:
        iteration += 1
        x -= lr * grad(x)
        x_1_values.append(x[0])
        x_2_values.append(x[1])
        obj_values.append(obj(x))
        print(f"{iteration:03d}, {x[0]:.3f}, {x[1]:.3f}, {obj(x):.3f}")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x_1_values, x_2_values, marker="o")
    ax[1].plot(obj_values)
    plt.savefig("outputs/gradient_descent.png")
