# 🧵 OS_Task1 – Multithreading Assignment

> Producer-Consumer in Java and 100 × 100 TensorFlow matrix multiplication with animation.

---

## 📌 Overview

This repository contains two programs:

1. **Producer-Consumer Problem using Java Threads**
2. **100 × 100 Matrix Multiplication using TensorFlow with Animation**

---

## 📂 Files in this Repository

| File | Description |
|------|-------------|
| `ProducerConsumer.java` | Producer-Consumer problem using Java threads |
| `matrix_multiplication.py` | 100 × 100 matrix multiplication using TensorFlow |
| `matrix_multiplication_100x100.mp4` | MP4 animation of matrix multiplication |
| `matrix_multiplication.gif` | GIF animation of matrix multiplication |
| `README.md` | Documentation for both programs |

---

# 1️⃣ Program 1 – Producer-Consumer Problem

## 📌 Description

The Producer-Consumer problem is implemented using **Java threads** with a shared circular buffer.

- The **Producer** adds items into the buffer.
- The **Consumer** removes items from the buffer.
- If the buffer is full, the Producer waits.
- If the buffer is empty, the Consumer waits.
- Synchronization is used to safely access the shared buffer.

The program uses a buffer size of **4** and produces and consumes **12 items**.

---

## 🔹 Concepts Used

- Java Threads
- Multithreading
- Synchronization
- Shared Buffer
- Circular Buffer
- `wait()`
- `notifyAll()`
- `join()`

---

## ⚙️ Approach

A fixed-size integer array is used as a circular buffer.

The program maintains:

- `in` → next insertion position
- `out` → next removal position
- `count` → number of items currently present in the buffer

A common lock object is used for synchronization.

When the buffer is full, the Producer waits.

When the buffer is empty, the Consumer waits.

After an item is produced or consumed, `notifyAll()` is used to wake the waiting thread.

---

## ▶️ How to Run Program 1

Run directly:

```bash
java ProducerConsumer.java
```

Or compile and run:

```bash
javac ProducerConsumer.java
java ProducerConsumer
```

---

## 🖥️ Sample Output

```text
Producer Consumer started

Buffer empty. Consumer waiting...

Produced 1 -> [1]
Consumed 1 -> []

Produced 2 -> [2]
Produced 3 -> [2, 3]
Produced 4 -> [2, 3, 4]

Produced 6 -> [3, 4, 5, 6]
Buffer full. Producer waiting...

Consumed 3 -> [4, 5, 6]

...

Producer Consumer finished
```

---

# 2️⃣ Program 2 – 100 × 100 Matrix Multiplication using TensorFlow

## 📌 Description

The second program performs multiplication of two matrices of size **100 × 100**.

The matrices are:

```text
Matrix A = 100 × 100
Matrix B = 100 × 100
Result Matrix C = 100 × 100
```

The operation is:

```text
Matrix A × Matrix B = Matrix C
```

TensorFlow is used to perform the matrix multiplication.

---

## 🔹 Concepts and Technologies Used

- Python
- TensorFlow
- NumPy
- Matrix Multiplication
- Matplotlib
- Animation
- FFmpeg

---

## ⚙️ Approach

Two random **100 × 100 matrices** are generated using NumPy.

The matrices are converted into TensorFlow tensors.

The multiplication is performed using:

```python
C = tf.matmul(A, B)
```

For each value in the resultant matrix, one row of Matrix A is multiplied with one column of Matrix B.

Conceptually:

```text
C[i][j] =
A[i][0] × B[0][j]
+
A[i][1] × B[1][j]
+
...
+
A[i][99] × B[99][j]
```

---

## 🎬 Animation

The animation displays the three matrices separately:

```text
Matrix A          Matrix B          Result Matrix C
100 × 100    ×    100 × 100    =      100 × 100
```

During the animation:

- 🔵 **Matrix A** – a horizontal black line indicates the current row.
- 🟢 **Matrix B** – a vertical black line indicates the current column.
- 🟠 **Matrix C** – the resultant matrix is filled progressively.

This visually demonstrates:

> **Row of Matrix A × Column of Matrix B → Result Matrix C**

---

## 🎥 Animation Output

The generated files are:

```text
matrix_multiplication_100x100.mp4
matrix_multiplication.gif
```

### GIF Preview

![Matrix Multiplication Animation](matrix_multiplication.gif)

---

## ▶️ Installation for Program 2

Python **3.10** is used.

### Create virtual environment

```bash
py -3.10 -m venv .venv
```

### Activate virtual environment

```bash
.\.venv\Scripts\Activate.ps1
```

### Install required libraries

```bash
pip install tensorflow numpy matplotlib imageio-ffmpeg
```

---

## ▶️ How to Run Program 2

```bash
python matrix_multiplication.py
```

---

## 🖥️ Sample Output

```text
TensorFlow Matrix Multiplication
--------------------------------

Matrix A shape: (100, 100)
Matrix B shape: (100, 100)
Result shape  : (100, 100)

First 5 x 5 values of Matrix A:
[...]

First 5 x 5 values of Matrix B:
[...]

First 5 x 5 values of Result Matrix:
[...]

Creating matrix multiplication video...

Result verified successfully.

Video file:
matrix_multiplication_100x100.mp4

Program completed.
```

---

# 🛠️ Technologies Used

## Program 1

- Java
- Java Threads
- Synchronization
- Circular Buffer

## Program 2

- Python
- TensorFlow
- NumPy
- Matplotlib
- FFmpeg
- Animation

---

# ✅ Conclusion

The first program demonstrates **thread synchronization using the Producer-Consumer problem in Java**.

The second program performs **100 × 100 matrix multiplication using TensorFlow** and provides an animated visualization of:

> **Matrix A × Matrix B = Result Matrix C**
