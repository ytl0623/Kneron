import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c
from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D

# modify the file name below

# depth result comparison
depth1_bin_path = "path/to/depth_file_1.bin"
depth2_bin_path = "path/to/depth_file_2.bin"
depth_out_path = "path/to/output_depth_diff.png"

# segmentation result comparison
seg1_bin_path = "path/to/segmentation_file_1.bin"
seg2_bin_path = "path/to/segmentation_file_2.bin"
seg_out_path = "path/to/output_seg_diff.png"

# read float depth result
depth_file_path = "path/to/depth_float_file.bin"

# draw disparity result
disparity_path = "path/to/hico_nir_rst_disparity.bin"

def compare_depth_binary(file1_path, file2_path, out_path):
    max_diff = -1
    max_diff_index = -1
    max_diff_x = 0
    max_diff_y = 0
    diffs = []
    i = 0
    max_1 = 0
    max_2 = 0
    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
        while True:
            byte1 = file1.read(4)
            byte2 = file2.read(4)

            if not byte1:
                break

            # Convert bytes to float values and calculate the difference
            val1 = struct.unpack("f", byte1)[0]
            val2 = struct.unpack("f", byte2)[0]
            diff = abs(val1 - val2)

            if diff > max_diff:
                max_diff = diff
                max_diff_index = i
                max_1 = val1
                max_2 = val2

            diffs.append(diff)

            i += 1

    if len(diffs) > 0:
        # Calculate the average and standard deviation of the differences
        avg_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        var_diff = np.var(diffs)
        min_diff = np.min(diffs)
        max_diff = np.max(diffs)
        print(f"Average difference: {avg_diff:.4f}")
        print(f"Standard deviation of difference: {std_diff:.4f}")
        print(f"variance: {var_diff:.4f}")
        print(f"min difference: {min_diff:.4f}")
        print(f"max difference: {max_diff:.4f}")
        max_diff_x = max_diff_index % 384
        max_diff_y = round(max_diff_index / 384)
        print(
            f"max difference is {max_diff} at ({max_diff_x}, {max_diff_y})., val1: {max_1:.4f} val2:{max_2:.4f}"
        )
        # Normalize the differences and apply a pseudo color map
        if std_diff > 0:
            norm_diffs = (diffs - avg_diff) / std_diff
        else:
            norm_diffs = diffs - avg_diff
        cmap = plt.cm.get_cmap("jet")
        rgba_diffs = cmap(norm_diffs)

        # Reshape the RGBA differences to a 2D array
        width = 384
        height = 128
        rgba_diffs = np.reshape(rgba_diffs, (height, width, 4))

        # Convert the PIL Image to a numpy array and display it with the color bar
        np_image = np.array(rgba_diffs)
        plt.imshow(np_image)
        plt.colorbar()
        plt.show()

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(height):
            for j in range(width):
                Z[i, j] = diffs[i * width + j]
        ax.plot_surface(X, Y, Z, cmap="jet")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Difference")
        plt.show()

        # Convert the NumPy array to a PIL Image object
        pil_image = Image.fromarray(np.uint8(np_image * 255))

        # Save the image to disk
        pil_image.save(out_path)
        print(f"Image saved to {out_path}.")
    else:
        print("Files are identical.")


def compare_segmentation_binary(file1_path, file2_path, out_path):
    diffs = []
    i = 0
    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
        while True:
            byte1 = file1.read(1)
            byte2 = file2.read(1)

            if not byte1:
                break

            if byte1 != byte2:
                diffs.append(255)
            else:
                diffs.append(0)

            i += 1

    if len(diffs) > 0:
        # Reshape the RGBA differences to a 2D array
        width = 400
        height = 300
        seg_diffs = np.reshape(diffs, (height, width, 1))

        # Convert the PIL Image to a numpy array and display it with the color bar
        np_image = np.array(seg_diffs)
        plt.imshow(np_image)
        plt.show()

        # Convert the NumPy array to a PIL Image object
        pil_image = Image.frombytes("L", (width, height), np_image.tobytes())

        # Save the image to disk
        pil_image.save(out_path)
        print(f"Image saved to {out_path}.")
    else:
        print("Files are identical.")


def read_single_depth_bin(file1_path):
    file1_data = []
    i = 0

    with open(file1_path, "rb") as file1:
        while True:
            byte1 = file1.read(4)
            if not byte1:
                break

            # Convert bytes to float values and calculate the difference
            val1 = struct.unpack("f", byte1)[0]

            file1_data.append(val1)

            i += 1
    print(i)

    # Reshape the RGBA differences to a 2D array
    width = 384
    height = 128

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(height):
        for j in range(width):
            k = height - i - 1
            Z[k, j] = file1_data[i * width + j]
    ax.plot_surface(X, Y, Z, cmap="jet")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")
    ax.set_title("3d image")
    plt.show()


def read_single_disparity_bin(disparity_path):
    file1_data = []
    i = 0

    with open(disparity_path, "rb") as file1:
        while True:
            byte1 = file1.read(1)
            if not byte1:
                break

            # Convert bytes to float values and calculate the difference
            val1 = struct.unpack("b", byte1)[0]

            file1_data.append(val1)

            i += 1
    print(i)

    # Reshape the RGBA differences to a 2D array
    width = 384
    height = 128

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(height):
        for j in range(width):
            k = height - i - 1
            Z[k, j] = file1_data[i * width + j]
    ax.plot_surface(X, Y, Z, cmap="jet")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")
    ax.set_title("disparity")
    plt.show()

# Uncomment the following line to perform depth comparison
# compare_depth_binary(depth1_bin_path, depth2_bin_path, depth_out_path)

# Uncomment the following line to perform segmentation comparison
# compare_segmentation_binary(seg1_bin_path, seg2_bin_path, seg_out_path)

# Uncomment the following lines to read and visualize a single depth binary file
# read_single_depth_bin(depth_file_path)

# Uncomment the following lines to read and visualize a single disparity binary file
# read_single_disparity_bin(disparity_path)
