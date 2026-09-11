import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle
import imageio_ffmpeg


N = 100
THREADS = 8

VIDEO_FILE = "matrix_multiplication_100x100.mp4"
GIF_FILE = "matrix_multiplication.gif"

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def build_matrices():

    random_gen = np.random.default_rng(10)

    matrix_a = random_gen.integers(
        1, 10, size=(N, N)
    )

    matrix_b = random_gen.integers(
        1, 10, size=(N, N)
    )

    tensor_a = tf.convert_to_tensor(
        matrix_a,
        dtype=tf.float32
    )

    tensor_b = tf.convert_to_tensor(
        matrix_b,
        dtype=tf.float32
    )

    return matrix_a, matrix_b, tensor_a, tensor_b


def calculate_cell(row_index, column_index, tensor_a, tensor_b):

    row = tensor_a[row_index]
    column = tensor_b[:, column_index]

    multiplied_values = row * column

    cell_value = tf.reduce_sum(
        multiplied_values
    )

    return row_index, column_index, float(cell_value.numpy())


def multiply_with_threads(tensor_a, tensor_b):

    result = np.zeros(
        (N, N),
        dtype=np.float32
    )

    finished_cells = []

    print("\nThreaded matrix multiplication started")
    print("--------------------------------------")
    print("Matrix size:", N, "x", N)
    print("Threads used:", THREADS)
    print("Total cell tasks:", N * N)

    with ThreadPoolExecutor(
        max_workers=THREADS
    ) as executor:

        tasks = []

        for row in range(N):

            for column in range(N):

                task = executor.submit(
                    calculate_cell,
                    row,
                    column,
                    tensor_a,
                    tensor_b
                )

                tasks.append(task)

        for task in as_completed(tasks):

            row, column, value = task.result()

            result[row, column] = value

            finished_cells.append(
                (row, column, value)
            )

    return result, finished_cells


def print_sample(matrix_a, matrix_b, result):

    print("\nMatrix multiplication completed")
    print("--------------------------------")

    print("A size:", matrix_a.shape)
    print("B size:", matrix_b.shape)
    print("C size:", result.shape)

    print("\nA sample:")
    print(matrix_a[:5, :5])

    print("\nB sample:")
    print(matrix_b[:5, :5])

    print("\nC sample:")
    print(result[:5, :5])


class MatrixAnimator:

    def __init__(
        self,
        matrix_a,
        matrix_b,
        result,
        finished_cells
    ):

        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.result = result
        self.finished_cells = finished_cells

        self.visible_result = np.full(
            (N, N),
            np.nan,
            dtype=np.float32
        )

        self.cells_in_one_frame = 50

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

        self.prepare_display()


    def prepare_display(self):

        self.ax_a.imshow(
            self.matrix_a,
            aspect="auto"
        )

        self.ax_a.set_title(
            "Matrix A\n100 × 100"
        )

        self.ax_a.set_xlabel("Columns")
        self.ax_a.set_ylabel("Rows")


        self.ax_b.imshow(
            self.matrix_b,
            aspect="auto"
        )

        self.ax_b.set_title(
            "Matrix B\n100 × 100"
        )

        self.ax_b.set_xlabel("Columns")
        self.ax_b.set_ylabel("Rows")


        self.result_image = self.ax_c.imshow(
            np.ma.masked_invalid(
                self.visible_result
            ),
            aspect="auto",
            vmin=np.min(self.result),
            vmax=np.max(self.result)
        )

        self.ax_c.set_title(
            "Result Matrix C\n100 × 100"
        )

        self.ax_c.set_xlabel("Columns")
        self.ax_c.set_ylabel("Rows")


        self.row_box = Rectangle(
            (-0.5, -0.5),
            N,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5
        )

        self.column_box = Rectangle(
            (-0.5, -0.5),
            1,
            N,
            fill=False,
            edgecolor="black",
            linewidth=2.5
        )

        self.result_box = Rectangle(
            (-0.5, -0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5
        )


        self.ax_a.add_patch(
            self.row_box
        )

        self.ax_b.add_patch(
            self.column_box
        )

        self.ax_c.add_patch(
            self.result_box
        )


        self.fig.suptitle(
            "100 × 100 Threaded TensorFlow Matrix Multiplication",
            fontsize=19,
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


        self.status_text = self.fig.text(
            0.5,
            0.11,
            "Preparing animation...",
            ha="center",
            fontsize=13,
            fontweight="bold"
        )

        self.cell_text = self.fig.text(
            0.5,
            0.06,
            "",
            ha="center",
            fontsize=11
        )


    def draw_frame(self, frame_number):

        start = (
            frame_number
            * self.cells_in_one_frame
        )

        end = min(
            start + self.cells_in_one_frame,
            len(self.finished_cells)
        )

        current_row = 0
        current_column = 0

        for index in range(start, end):

            row, column, value = self.finished_cells[
                index
            ]

            self.visible_result[
                row, column
            ] = value

            current_row = row
            current_column = column


        self.result_image.set_data(
            np.ma.masked_invalid(
                self.visible_result
            )
        )


        self.row_box.set_y(
            current_row - 0.5
        )

        self.column_box.set_x(
            current_column - 0.5
        )

        self.result_box.set_xy(
            (
                current_column - 0.5,
                current_row - 0.5
            )
        )


        self.status_text.set_text(
            f"Completed cells: {end} / {N * N}"
        )

        self.cell_text.set_text(
            f"Current cell: "
            f"C[{current_row}][{current_column}]"
        )


        return (
            self.result_image,
            self.row_box,
            self.column_box,
            self.result_box,
            self.status_text,
            self.cell_text
        )


    def create_animation(self):

        frame_count = int(
            np.ceil(
                len(self.finished_cells)
                / self.cells_in_one_frame
            )
        )


        animation = FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=frame_count,
            interval=100,
            repeat=False
        )


        print("\nCreating MP4 animation...")

        video_writer = FFMpegWriter(
            fps=10
        )

        animation.save(
            VIDEO_FILE,
            writer=video_writer,
            dpi=120
        )

        print("MP4 created successfully.")


        self.visible_result[:] = np.nan


        print("\nCreating GIF animation...")

        gif_writer = PillowWriter(
            fps=10
        )

        animation.save(
            GIF_FILE,
            writer=gif_writer,
            dpi=90
        )

        print("GIF created successfully.")

        plt.close(self.fig)


def main():

    matrix_a, matrix_b, tensor_a, tensor_b = build_matrices()


    threaded_result, finished_cells = multiply_with_threads(
        tensor_a,
        tensor_b
    )


    print_sample(
        matrix_a,
        matrix_b,
        threaded_result
    )


    reference_result = tf.matmul(
        tensor_a,
        tensor_b
    ).numpy()


    if np.allclose(
        threaded_result,
        reference_result
    ):

        print(
            "\nThreaded result verified successfully."
        )

    else:

        print(
            "\nResult verification failed."
        )


    print(
        "Completed threaded cell tasks:",
        len(finished_cells)
    )


    animator = MatrixAnimator(
        matrix_a,
        matrix_b,
        threaded_result,
        finished_cells
    )

    animator.create_animation()


    print("\nGenerated files:")
    print(VIDEO_FILE)
    print(GIF_FILE)

    print("\nProgram completed.")


if __name__ == "__main__":
    main()
