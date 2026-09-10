import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
import imageio_ffmpeg


N = 100
OUTPUT_VIDEO = "matrix_multiplication_100x100.mp4"

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def build_matrices():
    rng = np.random.default_rng(10)

    a_values = rng.integers(
        1, 10, size=(N, N)
    )

    b_values = rng.integers(
        1, 10, size=(N, N)
    )

    a_tensor = tf.convert_to_tensor(
        a_values,
        dtype=tf.float32
    )

    b_tensor = tf.convert_to_tensor(
        b_values,
        dtype=tf.float32
    )

    return a_values, b_values, a_tensor, b_tensor


def print_sample(a, b, c):
    print("\nMatrix multiplication using TensorFlow")
    print("--------------------------------------")

    print("A size:", a.shape)
    print("B size:", b.shape)
    print("C size:", c.shape)

    print("\nA sample:")
    print(a[:5, :5])

    print("\nB sample:")
    print(b[:5, :5])

    print("\nC sample:")
    print(c[:5, :5])


class MatrixAnimator:

    def __init__(self, a_np, b_np, a_tf, b_tf, c_np):

        self.a_np = a_np
        self.b_np = b_np
        self.a_tf = a_tf
        self.b_tf = b_tf
        self.c_np = c_np

        self.partial = np.zeros(
            (N, N),
            dtype=np.float32
        )

        self.fig, axes = plt.subplots(
            1, 3, figsize=(16, 7)
        )

        self.ax_a = axes[0]
        self.ax_b = axes[1]
        self.ax_c = axes[2]

        self.fig.subplots_adjust(
            top=0.80,
            bottom=0.20,
            wspace=0.35
        )

        self.prepare_axes()


    def prepare_axes(self):

        self.ax_a.imshow(
            self.a_np,
            aspect="auto"
        )

        self.ax_a.set_title(
            "Matrix A\n100 × 100"
        )

        self.ax_a.set_xlabel("Columns")
        self.ax_a.set_ylabel("Rows")


        self.ax_b.imshow(
            self.b_np,
            aspect="auto"
        )

        self.ax_b.set_title(
            "Matrix B\n100 × 100"
        )

        self.ax_b.set_xlabel("Columns")
        self.ax_b.set_ylabel("Rows")


        self.result_view = self.ax_c.imshow(
            self.partial,
            aspect="auto",
            vmin=0,
            vmax=np.max(self.c_np)
        )

        self.ax_c.set_title(
            "Result Matrix C\n100 × 100"
        )

        self.ax_c.set_xlabel("Columns")
        self.ax_c.set_ylabel("Rows")


        self.a_marker = Rectangle(
            (-0.5, -0.5),
            N,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5
        )

        self.b_marker = Rectangle(
            (-0.5, -0.5),
            1,
            N,
            fill=False,
            edgecolor="black",
            linewidth=2.5
        )

        self.c_marker = Rectangle(
            (-0.5, -0.5),
            N,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5
        )

        self.ax_a.add_patch(self.a_marker)
        self.ax_b.add_patch(self.b_marker)
        self.ax_c.add_patch(self.c_marker)


        self.fig.suptitle(
            "100 × 100 TensorFlow Matrix Multiplication",
            fontsize=20,
            fontweight="bold"
        )

        self.fig.text(
            0.345,
            0.50,
            "×",
            fontsize=38,
            fontweight="bold",
            ha="center"
        )

        self.fig.text(
            0.665,
            0.50,
            "=",
            fontsize=38,
            fontweight="bold",
            ha="center"
        )


        self.status = self.fig.text(
            0.5,
            0.11,
            "Preparing animation...",
            ha="center",
            fontsize=14,
            fontweight="bold"
        )

        self.detail = self.fig.text(
            0.5,
            0.06,
            "",
            ha="center",
            fontsize=12
        )


    def draw_frame(self, step):

        row_index = step

        selected_row = self.a_tf[
            row_index:row_index + 1
        ]

        computed_row = tf.matmul(
            selected_row,
            self.b_tf
        )

        self.partial[
            row_index
        ] = computed_row.numpy()[0]

        self.result_view.set_data(
            self.partial
        )

        self.a_marker.set_y(
            row_index - 0.5
        )

        self.b_marker.set_x(
            row_index - 0.5
        )

        self.c_marker.set_y(
            row_index - 0.5
        )

        self.status.set_text(
            f"Step {row_index + 1} / {N}"
        )

        self.detail.set_text(
            f"Using row {row_index + 1} from A "
            f"with columns of B to form row {row_index + 1} of C"
        )

        return (
            self.result_view,
            self.a_marker,
            self.b_marker,
            self.c_marker,
            self.status,
            self.detail
        )


    def save_video(self):

        animation = FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=N,
            interval=150,
            repeat=False
        )

        writer = FFMpegWriter(
            fps=10
        )

        print("\nCreating animation...")

        animation.save(
            OUTPUT_VIDEO,
            writer=writer,
            dpi=120
        )

        plt.close(self.fig)


def main():

    a_np, b_np, a_tf, b_tf = build_matrices()

    c_tf = tf.matmul(
        a_tf,
        b_tf
    )

    c_np = c_tf.numpy()

    print_sample(
        a_np,
        b_np,
        c_np
    )

    animator = MatrixAnimator(
        a_np,
        b_np,
        a_tf,
        b_tf,
        c_np
    )

    animator.save_video()

    if np.allclose(
        animator.partial,
        c_np
    ):
        print("\nResult verified successfully.")
    else:
        print("\nResult verification failed.")

    print("\nVideo file:")
    print(OUTPUT_VIDEO)


if __name__ == "__main__":
    main()