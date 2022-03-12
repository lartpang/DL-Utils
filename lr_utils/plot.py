import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tv_tf
from scipy import signal
from torchvision.utils import make_grid

MARKERS = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
]


@torch.no_grad()
def plot_results(data_container, save_path=None):
    """Plot the results conresponding to the batched images based on the `make_grid` method from `torchvision`.

    Args:
        data_container (dict): Dict containing data you want to plot.
        save_path (str): Path of the exported image.
    """
    axes = plt.subplots(nrows=len(data_container), ncols=1)[1].ravel()
    plt.subplots_adjust(hspace=0.03, left=0.05, bottom=0.01, right=0.99, top=0.99)

    for subplot_id, (name, data) in enumerate(data_container.items()):
        grid = make_grid(data, nrow=data.shape[0], padding=2, normalize=False)
        grid_image = np.asarray(tv_tf.to_pil_image(grid))
        axes[subplot_id].imshow(grid_image)
        axes[subplot_id].set_ylabel(name)
        axes[subplot_id].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_lr_coef_curve(lr_lambda, num_steps, save_path=None):
    fig, ax = plt.subplots()

    # give plot a title
    ax.set_title("Learning Rate Coefficient Curve")
    # make axis labels
    ax.set_xlabel("Index")
    ax.set_ylabel("Coefficient")
    # set ticks
    ax.set_xticks(np.linspace(0, num_steps, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    # set lim
    ax.set_xlim((-int(num_steps * 0.1), int(num_steps * 1.5)))
    ax.set_ylim((-0.1, 1))

    x_data = np.arange(num_steps)
    y_data = np.array([lr_lambda(x) for x in x_data])

    ax.plot(x_data, y_data, linewidth=2)

    maximum_xs = signal.argrelextrema(y_data, comparator=np.greater_equal)[0]
    maximum_ys = y_data[maximum_xs]
    minimum_xs = signal.argrelextrema(y_data, comparator=np.less_equal)[0]
    minimum_ys = y_data[minimum_xs]

    end_point_xs = np.array([x_data[0], x_data[-1]])
    end_point_ys = np.array([y_data[0], y_data[-1]])
    for pt in zip(
        np.concatenate((maximum_xs, minimum_xs, end_point_xs)),
        np.concatenate((maximum_ys, minimum_ys, end_point_ys)),
    ):
        ax.text(pt[0], pt[1], s=f"x={pt[0]:d}")
        ax.text(pt[0], pt[1] - 0.05, s=f"y={pt[1]:.3e}")

    if save_path:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_lr_curve_for_scheduler(scheduler, num_steps, save_path=None):
    scheduler = copy.deepcopy(scheduler)

    plt.rc("xtick", labelsize="small")
    plt.rc("ytick", labelsize="small")

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 4), dpi=600)
    # give plot a title
    ax.set_title("Learning Rate Curve")
    # make axis labels
    ax.set_xlabel("Iter")
    ax.set_ylabel("LR")

    x_data = np.arange(num_steps)
    ys = []
    for _ in x_data:
        ys.append(max(scheduler.get_last_lr()))
        scheduler.step()
    y_data = np.array(ys)

    # set lim
    x_min, x_max = 0, num_steps - 1
    dx = num_steps * 0.1
    ax.set_xlim(x_min - dx, x_max + 2 * dx)

    y_min, y_max = y_data.min(), y_data.max()
    dy = (y_data.max() - y_data.min()) * 0.1
    ax.set_ylim((y_min - dy, y_max + dy))

    marker_on = [0, -1]
    key_point_xs = [0, num_steps - 1]
    for idx in range(1, len(y_data) - 1):
        prev_y = y_data[idx - 1]
        curr_y = y_data[idx]
        next_y = y_data[idx + 1]
        if (
            (curr_y > prev_y and curr_y >= next_y)
            or (curr_y >= prev_y and curr_y > next_y)
            or (curr_y <= prev_y and curr_y < next_y)
            or (curr_y < prev_y and curr_y <= next_y)
        ):
            marker_on.append(idx)
            key_point_xs.append(idx)

    marker_on = sorted(set(marker_on))
    key_point_xs = sorted(set(key_point_xs))
    key_point_ys = []
    for x in key_point_xs:
        y = y_data[x]
        key_point_ys.append(y)
        ax.annotate(
            text=f"({x:d},{y:.3e})",
            xy=(x, y),
            xytext=(-10, +10),
            size="small",
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"),
            rotation='vertical',
        )

    # set ticks
    ax.set_xticks(key_point_xs)
    # ax.set_yticks(key_point_ys)

    ax.plot(x_data, y_data, marker="o", markevery=marker_on)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    plt.tight_layout()
    plt.savefig(save_path)


def plot_lr_curve(log_path):
    lrs = []
    with open(log_path, encoding="utf-8", mode="r") as f:
        for line in f:
            if "Lr:" not in line:
                continue
            line = line.rstrip()
            # [Train@0/13160 0/329 0/40] | Lr:[0.0005, 0.0005] | M:0.65033/C:0.65033 | [32, 3, 384, 384] | bce: 0.650327
            lrs.append([float(f) for f in line.split(" | ")[1][4:-1].split(", ")])

    _, ax = plt.subplots()

    # give plot a title
    ax.set_title("Learning Rate Curve")
    # make axis labels
    ax.set_xlabel("Index")
    ax.set_ylabel("LR")
    # set ticks
    ax.set_xticks(np.linspace(0, len(lrs), 11))
    ax.set_yticks(np.linspace(0, 0.1, 11))
    # set lim
    ax.set_xlim((-int(len(lrs) * 0.1), int(len(lrs) * 1.5)))
    ax.set_ylim((-0.01, 0.1))

    x_data = np.arange(len(lrs))
    for y_idx, y_data in enumerate(zip(*lrs)):
        y_data = np.array(y_data)
        print(y_data)
        ax.plot(x_data, y_data, linewidth=1, label=str(y_idx), marker=MARKERS[y_idx])

    maximum_xs = signal.argrelextrema(y_data, comparator=np.greater_equal)[0]
    maximum_ys = y_data[maximum_xs]
    minimum_xs = signal.argrelextrema(y_data, comparator=np.less_equal)[0]
    minimum_ys = y_data[minimum_xs]

    end_point_xs = np.array([x_data[0], x_data[-1]])
    end_point_ys = np.array([y_data[0], y_data[-1]])
    for pt in zip(
        np.concatenate((maximum_xs, minimum_xs, end_point_xs)),
        np.concatenate((maximum_ys, minimum_ys, end_point_ys)),
    ):
        ax.text(pt[0], pt[1], s=f"x={pt[0]:d}")
        ax.text(pt[0], pt[1] - 0.005, s=f"y={pt[1]:.3e}")

    ax.legend()
    plt.show()
    
