# Depth and Segmentation Comparison Python Script
# Depth and Disparity Reader Python Script

## Description
This Python script performs comparisons between binary files containing depth data and segmentation data. It visualizes the differences between two depth files and two segmentation files, displaying the results as images and 3D plots.

## Dependencies
- struct
- numpy
- matplotlib
- PIL (Python Imaging Library)

## Usage Instructions

### Depth Result Comparison
1. Modify the following variables to specify the paths of the binary files for depth comparison:
   - `depth1_bin_path`: Path to the first depth binary file.
   - `depth2_bin_path`: Path to the second depth binary file.
   - `depth_out_path`: Path to save the output image showing the differences.

2. Uncomment the function call `compare_depth_binary(depth1_bin_path, depth2_bin_path, depth_out_path)`.

### Segmentation Result Comparison (Optional)
1. If you want to compare segmentation results, modify the following variables to specify the paths of the binary files:
   - `seg1_bin_path`: Path to the first segmentation binary file.
   - `seg2_bin_path`: Path to the second segmentation binary file.
   - `seg_out_path`: Path to save the output image showing the differences.

2. Uncomment the function call `compare_segmentation_binary(seg1_bin_path, seg2_bin_path, seg_out_path)`.

### Read Single Depth Binary (Optional)
1. If you want to read and visualize a single depth binary file, modify the `depth_file_path` variable to specify the path of the binary file.

2. Uncomment the function call `read_single_depth_bin(depth_file_path)`.

### Read Single Disparity Binary (Optional)
1. If you want to read and visualize a single disparity binary file, modify the `disparity_path` variable to specify the path of the binary file.

2. Uncomment the function call `read_single_disparity_bin(disparity_path)`.

## Example
Here's an example of how to use the script with specific file paths:

```python
# depth result comparison
depth1_bin_path = "path/to/depth_file_1.bin"
depth2_bin_path = "path/to/depth_file_2.bin"
depth_out_path = "path/to/output_depth_diff.png"

# Uncomment the following line to perform depth comparison
# compare_depth_binary(depth1_bin_path, depth2_bin_path, depth_out_path)

# segmentation result comparison
seg1_bin_path = "path/to/segmentation_file_1.bin"
seg2_bin_path = "path/to/segmentation_file_2.bin"
seg_out_path = "path/to/output_seg_diff.png"

# Uncomment the following line to perform segmentation comparison
# compare_segmentation_binary(seg1_bin_path, seg2_bin_path, seg_out_path)

# Uncomment the following lines to read and visualize a single depth binary file
# depth_file_path = "path/to/single_depth_file.bin"
# read_single_depth_bin(depth_file_path)

# Uncomment the following lines to read and visualize a single disparity binary file
# disparity_path = "path/to/single_disparity_file.bin"
# read_single_disparity_bin(disparity_path)
