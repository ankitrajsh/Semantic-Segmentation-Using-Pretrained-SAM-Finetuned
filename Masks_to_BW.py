

from PIL import Image
import os
import colorsys

def is_green(pixel):
    # Convert RGB to HSV
    r, g, b = pixel[:3]
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    # Define green in HSV: broader range for hue, and reasonable ranges for saturation and value
    return (0.25 <= h <= 0.45) and s >= 0.3 and v >= 0.2

def convert_green_masks(directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all the files in the input directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Assuming the masks are in PNG format
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Open the image
            img = Image.open(file_path).convert('RGB')  # Ensure it's in RGB

            # Create a new image to store the black and white version
            bw_img = Image.new('1', img.size)  # '1' for 1-bit pixels, black and white

            # Process the image to set green pixels to white and others to black
            pixels = img.load()
            bw_pixels = bw_img.load()
            for i in range(img.width):
                for j in range(img.height):
                    if is_green(pixels[i, j]):
                        bw_pixels[i, j] = 255  # Set to white
                    else:
                        bw_pixels[i, j] = 0  # Set to black

            # Save the converted image
            bw_img.save(output_path)
            print(f'Processed {filename}')

# Directory paths
input_directory = 'data2/VOC_level_green_patch'
output_directory = 'data2/VOC_level_green_patch_classvalue11'

# Convert masks
convert_green_masks(input_directory, output_directory)
print("Conversion complete!")
