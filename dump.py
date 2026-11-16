import matplotlib.pyplot as plt
import numpy as np


# mx = []
# for y in range(10):
#     for x in range(10):
#         mx[y][x] = x+y

# # Matrix filled with zeros
# matrix = [[0 for _ in range(10)] for _ in range(10)]

# # Matrix filled with consecutive numbers (0 to 99)
# matrix_seq = [[i * 10 + j for j in range(10)] for i in range(10)]

# print(matrix_seq)



# Create a 10x10 matrix
matrix = np.zeros((10, 10), dtype=int)
matrix[3][8] = 10
# matrix = np.random.rand(10, 10)

fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(matrix, cmap="Blues")

# plt.figure(figsize=(7, 7))

# Set custom colormap (e.g., 'magma', 'coolwarm', 'plasma', 'YlGnBu')
# Perceptually Uniform: "viridis", "magma", "plasma", "inferno"
# Diverging: "coolwarm", "bwr", "seismic", "Spectral"
# Sequential: "Blues", "YlGnBu", "Greens", "Purples"
plt.imshow(matrix, cmap="Blues")

# Annotate each cell with its value
for i in range(10):
    for j in range(10):
        val = matrix[i, j]
        # Adjust text color for readability against dark/light background
        # color = "white" if val > 50 else "black"
        color = "black"
        ax.text(j, i, str(val), ha="center", va="center", color=color)

# Add colorbar and labels
plt.colorbar(label="Intensity Value")
plt.title("10x10 Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Ensure integer tick marks from 0 to 9
plt.xticks(range(10))
plt.yticks(range(10))

plt.show()