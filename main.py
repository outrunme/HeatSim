import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time

starttime = time.time()

counter = 0
counter2 = 0
frames = 1000
time_resolution = 0.01
interval = 20
simulation_time = 300
time_steps = int(np.ceil(simulation_time / time_resolution))
if frames > time_steps:
    frames = time_steps
x_resolution = 0.5
y_resolution = 0.5
conductivity = 1

x = np.arange(0, 100, step=x_resolution)
y = np.arange(0, 100, step=y_resolution)
heatmap = np.zeros((len(x), len(y)))
heatmap[:, 0:1] = np.ones((len(x), 1))
heatmap[:, len(y) - 1 : len(y)] = np.zeros((len(x), 1))
heatmap[0:1, :] = np.ones((1, len(y)))
heatmap[len(x) - 1 : len(x), :] = np.zeros((1, len(y)))
state = np.zeros((frames, len(x), len(y)))
heatmap_derivative2_x = np.zeros_like(heatmap)
heatmap_derivative2_y = np.zeros_like(heatmap)
xx_derivative = np.zeros((len(x), len(y)))
yy_derivative = np.zeros((len(x), len(y)))


def compute_derivative_x(i, j, function, x_diff):
    return (function[i + 1, j] - 2 * function[i, j] + function[i - 1, j]) / x_diff**2


def compute_derivative_y(i, j, function, y_diff):
    return (function[i, j + 1] - 2 * function[i, j] + function[i, j - 1]) / y_diff**2


# Create meshgrid for indexing
X, Y = np.meshgrid(
    range(1, len(x) - 1),
    range(1, len(y) - 1),
)

# Compute derivatives and update heatmap
for k in range(time_steps):
    # Compute derivatives using vectorized operations
    heatmap_derivative2_x[1:-1, 1:-1] = (
        heatmap[X + 1, Y] - 2 * heatmap[X, Y] + heatmap[X - 1, Y]
    ) / x_resolution**2

    heatmap_derivative2_y[1:-1, 1:-1] = (
        heatmap[X, Y + 1] - 2 * heatmap[X, Y] + heatmap[X, Y - 1]
    ) / y_resolution**2

    # Update heatmap
    heatmap += (
        time_resolution
        * conductivity
        * (heatmap_derivative2_x.T + heatmap_derivative2_y.T)
    )

    counter += 1
    if (counter * frames * time_resolution / simulation_time) >= 1:
        state[counter2] = heatmap
        counter = 0
        counter2 += 1


endtime = time.time()
time_taken = endtime - starttime
print(f"Time taken for simulation: {time_taken}")

# Set up the figure and axis
fig, ax = plt.subplots()


heatmap_img = ax.imshow(
    heatmap,
    cmap="plasma",
    interpolation="nearest",
)

fig.colorbar(heatmap_img, ax=ax)


def update(frame):
    heatmap_img.set_data(state[frame, :, :])
    return (heatmap_img,)


ani = animation.FuncAnimation(
    fig, update, frames=range(frames), blit=True, repeat=True, interval=interval
)

# Set labels and title
ax.set_title("Animated Heatmap")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")

plt.show()
