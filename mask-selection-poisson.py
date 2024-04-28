import tkinter as tk
import tkinter.filedialog
from PIL import Image
import numpy as np
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson
from pyamg import ruge_stuben_solver
import matplotlib.pyplot as plt
from skimage.draw import polygon

# Function to get the path of an image file selected by the user through a file dialog.
def get_image_path_from_user(prompt_message):
    # Create a Tkinter window and hide it immediately.
    root = tkinter.Tk()
    root.withdraw()
    
    # Open a file dialog to allow the user to select an image file.
    file_path = tkinter.filedialog.askopenfilename(title=prompt_message)
    
    # Return the selected file path.
    return file_path


def rgb_to_gray_mat(image_path):
    # Open the image and convert it to grayscale.
    grayscale_image = Image.open(image_path).convert("L")
    
    # Convert the grayscale image to a NumPy array.
    grayscale_array = np.asarray(grayscale_image)
    
    # Return the grayscale NumPy array.
    return grayscale_array


# Function to get an image path from the user and split it into RGB channels.
def get_image_from_user(message, source_shape=(0, 0)):
    # Get the image path from the user.
    image_path = get_image_path_from_user(message)
    
    # Split the image into RGB channels.
    rgb_channels = split_image_to_rgb(image_path)
    
    # Check if the dimensions of the source image are smaller than the dimensions of the destination image.
    if not np.all(np.asarray(source_shape) < np.asarray(rgb_channels[0].shape)):
        # If the dimensions are not as expected, prompt the user again with a specific message.
        return get_image_from_user(
            "Open destination image with resolution bigger than "
            + str(tuple(np.asarray(source_shape) + 1)),
            source_shape,
        )
    
    # Return the image path and the RGB channels.
    return image_path, rgb_channels


# Function to create a binary mask based on a polygon inscribed by the user on a grayscale image.
def poly_mask(image_path, number_of_points=100):
    # Convert the image to grayscale.
    img = rgb_to_gray_mat(image_path)
    
    # Display the image and prompt the user to inscribe a polygon.
    plt.figure("source image")
    plt.title("Inscribe the region you would like to blend inside a polygon")
    plt.imshow(img, cmap="gray")
    
    # Get points from user input.
    points = np.asarray(plt.ginput(number_of_points, timeout=-1))
    
    # Close the plot.
    plt.close("all")
    
    # Check if the user inscribed at least three points.
    if len(points) < 3:
        # If not, create a mask covering the entire image.
        min_row, min_col = (0, 0)
        max_row, max_col = img.shape
        mask = np.ones(img.shape)
    else:
        # Flip the points to match the image coordinates.
        points = np.fliplr(points)
        
        # Get the row and column indices of the points inside the polygon.
        in_poly_row, in_poly_col = polygon(tuple(points[:, 0]), tuple(points[:, 1]), img.shape)
        
        # Calculate the bounding box of the polygon.
        min_row, min_col = np.max(
            np.vstack(
                [np.floor(np.min(points, axis=0)).astype(int).reshape((1, 2)), (0, 0)]
            ),
            axis=0,
        )
        max_row, max_col = np.min(
            np.vstack(
                [np.ceil(np.max(points, axis=0)).astype(int).reshape((1, 2)), img.shape]
            ),
            axis=0,
        )
        
        # Create a binary mask.
        mask = np.zeros(img.shape)
        mask[in_poly_row, in_poly_col] = 1
        mask = mask[min_row:max_row, min_col:max_col]
    
    # Return the mask and the bounding box coordinates.
    return mask, min_row, max_row, min_col, max_col



# Function to split an RGB image into its red, green, and blue channels.
def split_image_to_rgb(image_path):
    # Open the image and split it into its RGB channels.
    red_channel, green_channel, blue_channel = Image.Image.split(Image.open(image_path))
    
    # Convert the channels to NumPy arrays and return them.
    return np.asarray(red_channel), np.asarray(green_channel), np.asarray(blue_channel)


# Function to crop each channel of an RGB image based on specified row and column limits.
def crop_image_by_limits(src, min_row, max_row, min_col, max_col):
    # Unpack the RGB channels from the source.
    r, g, b = src
    
    # Crop each channel based on the specified limits.
    r = r[min_row:max_row, min_col:max_col]
    g = g[min_row:max_row, min_col:max_col]
    b = b[min_row:max_row, min_col:max_col]
    
    # Return the cropped channels.
    return r, g, b


# Function to adjust the coordinates of a corner point to keep it within the boundaries of a destination image.
def keep_src_in_dst_boundaries(corner, dst_shape, src_shape):
    # Iterate over each dimension of the corner point.
    for idx in range(len(corner)):
        # Ensure the coordinate is not less than 1.
        if corner[idx] < 1:
            corner[idx] = 1
        # Ensure the coordinate is not greater than the difference between the destination shape and source shape.
        if corner[idx] > dst_shape[idx] - src_shape[idx] - 1:
            corner[idx] = dst_shape[idx] - src_shape[idx] - 1
    
    # Return the adjusted corner coordinates.
    return corner


# Function to determine the top-left corner of the source image on the destination image.
def top_left_corner_of_src_on_dst(dst_img_path, src_shape):
    # Convert the destination image to grayscale.
    gry_dst = rgb_to_gray_mat(dst_img_path)
    
    # Display the destination image and prompt the user to select a point.
    plt.figure("destination image")
    plt.title("Where would you like to blend it..?")
    plt.imshow(gry_dst, cmap="gray")
    
    # Get the user's input for the center point.
    center = np.asarray(plt.ginput(2, -1, True)).astype(int)
    
    # Close the plot.
    plt.close("all")
    
    # Check if the user selected a point.
    if len(center) < 1:
        # If no point was selected, default to the center of the destination image.
        center = np.asarray([[gry_dst.shape[1] // 2, gry_dst.shape[0] // 2]]).astype(int)
    elif len(center) > 1:
        # If multiple points were selected, use the first one.
        center = np.asarray([center[0]])
    
    # Calculate the top-left corner of the source image on the destination image.
    corner = [center[0][1] - src_shape[0] // 2, center[0][0] - src_shape[1] // 2]
    
    # Adjust the corner to ensure it stays within the boundaries of the destination image.
    return keep_src_in_dst_boundaries(corner, gry_dst.shape, src_shape)


# Function to crop a portion of the destination image under the source image.
def crop_dst_under_src(dst_img, corner, src_shape):
    # Crop the portion of the destination image under the source image.
    dst_under_src = dst_img[
        corner[0] : corner[0] + src_shape[0], corner[1] : corner[1] + src_shape[1]
    ]
    return dst_under_src


# Function to calculate the Laplacian matrix of an array.
def laplacian(array):
    return (
        poisson(array.shape, format="csr")
        * csr_matrix(array.flatten()).transpose().toarray()
    )


# Function to set boundary conditions based on the portion of the destination image under the source image.
def set_boundary_condition(b, dst_under_src):
    # Set boundary conditions.
    b[1, :] = dst_under_src[1, :]
    b[-2, :] = dst_under_src[-2, :]
    b[:, 1] = dst_under_src[:, 1]
    b[:, -2] = dst_under_src[:, -2]
    
    # Remove the boundary pixels.
    b = b[1:-1, 1:-1]
    
    return b


# Function to construct the constant vector used in solving linear systems of equations.
def construct_const_vector(mask, mixed_grad, dst_under_src, src_laplacianed, src_shape):
    # Compute Laplacian of the destination under the source.
    dst_laplacianed = laplacian(dst_under_src)
    
    # Reshape the input arrays.
    b = np.reshape(
        (1 - mixed_grad) * mask * np.reshape(src_laplacianed, src_shape)
        + mixed_grad * mask * np.reshape(dst_laplacianed, src_shape)
        + (mask - 1) * (-1) * np.reshape(dst_laplacianed, src_shape),
        src_shape,
    )
    
    # Set boundary conditions for the constant vector.
    return set_boundary_condition(b, dst_under_src)


# Function to fix coefficients under boundary conditions.
def fix_coeff_under_boundary_condition(coeff, shape):
    shape_prod = np.prod(np.asarray(shape))
    arange_space = np.arange(shape_prod).reshape(shape)
    arange_space[1:-1, 1:-1] = -1
    index_to_change = arange_space[arange_space > -1]
    for j in index_to_change:
        coeff[j, j] = 1
        if j - 1 > -1:
            coeff[j, j - 1] = 0
        if j + 1 < shape_prod:
            coeff[j, j + 1] = 0
        if j - shape[-1] > -1:
            coeff[j, j - shape[-1]] = 0
        if j + shape[-1] < shape_prod:
            coeff[j, j + shape[-1]] = 0
    return coeff


# Function to construct the coefficient matrix used in solving linear systems of equations.
def construct_coefficient_matrix(shape):
    # Construct the Poisson matrix.
    a = poisson(shape, format="lil")
    
    # Fix coefficients under boundary conditions.
    a = fix_coeff_under_boundary_condition(a, shape)
    
    return a


# Function to build the linear system of equations for image blending.
def build_linear_system(mask, src_img, dst_under_src, mixed_grad):
    # Compute Laplacian of the source image.
    src_laplacianed = laplacian(src_img)
    
    # Construct constant vector and coefficient matrix.
    b = construct_const_vector(mask, mixed_grad, dst_under_src, src_laplacianed, src_img.shape)
    a = construct_coefficient_matrix(b.shape)
    
    return a, b


# Function to solve the linear system of equations for image blending.
def solve_linear_system(a, b, b_shape):
    # Solve the linear system using Ruge-Stuben solver.
    multi_level = ruge_stuben_solver(csr_matrix(a))
    x = np.reshape((multi_level.solve(b.flatten(), tol=1e-10)), b_shape)
    
    # Clip pixel values to [0, 255] range.
    x[x < 0] = 0
    x[x > 255] = 255
    
    return x


# Function to blend a patch into a destination image at a specified corner.
def blend(dst, patch, corner, patch_shape, blended):
    # Create a copy of the destination image.
    mixed = dst.copy()
    
    # Overlay the patch onto the destination image at the specified corner.
    mixed[
        corner[0] : corner[0] + patch_shape[0], corner[1] : corner[1] + patch_shape[1]
    ] = patch
    
    # Convert the blended image back to PIL format and append it to the blended list.
    blended.append(Image.fromarray(mixed))
    
    return blended


# Function to perform Poisson and naive blending.
def poisson_and_naive_blending(mask, corner, src_rgb, dst_rgb, mixed_grad):
    poisson_blended = []
    naive_blended = []
    for color in range(3):
        # Extract the source and destination images for the current color channel.
        src = src_rgb[color]
        dst = dst_rgb[color]
        
        # Crop the portion of the destination image under the source image.
        dst_under_src = crop_dst_under_src(dst, corner, src.shape)
        
        # Build linear system of equations and solve it for Poisson blending.
        a, b = build_linear_system(mask, src, dst_under_src, mixed_grad)
        x = solve_linear_system(a, b, b.shape)
        
        # Blend the result into the destination image.
        poisson_blended = blend(
            dst, x, (corner[0] + 1, corner[1] + 1), b.shape, poisson_blended
        )
        
        # Perform naive blending.
        crop_src = mask * src + (mask - 1) * (-1) * dst_under_src
        naive_blended = blend(dst, crop_src, corner, src.shape, naive_blended)
    
    return poisson_blended, naive_blended


# Function to merge, save, and display the blended images.
def merge_save_show(split_img, img_title):
    # Merge the split images into an RGB image.
    merged = Image.merge("RGB", tuple(split_img))
    
    # Save the merged image.
    merged.save(img_title + ".png")
    
    # Show the merged image.
    merged.show(img_title)


def main():
    # Open the source image and get the RGB channels.
    src_img_path, src_rgb = get_image_from_user("Open source image")

    # Create a polygonal mask based on user input.
    mask, *mask_limits = poly_mask(src_img_path)

    # Crop the source RGB channels based on the polygonal mask limits.
    src_rgb_cropped = crop_image_by_limits(src_rgb, *mask_limits)

    # Open the destination image and get its RGB channels.
    dst_img_path, dst_rgb = get_image_from_user("Open destination image", src_rgb_cropped[0].shape)

    # Determine the top-left corner of the source image on the destination image.
    corner = top_left_corner_of_src_on_dst(dst_img_path, src_rgb_cropped[0].shape)

    # Blend the images using both Poisson and naive blending techniques.
    poisson_blended, naive_blended = poisson_and_naive_blending(mask, corner, src_rgb_cropped, dst_rgb, 0.3)

    # Merge, save, and show the results of naive blending.
    merge_save_show(naive_blended, "Naive Blended")

    # Merge, save, and show the results of Poisson blending.
    merge_save_show(poisson_blended, "Poisson Blended")



if __name__ == "__main__":
    main()
