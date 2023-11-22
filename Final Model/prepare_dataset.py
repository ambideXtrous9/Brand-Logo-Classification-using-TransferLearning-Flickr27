from PIL import Image,UnidentifiedImageError
import os

def crop_and_save(image_path, coordinates, output_folder):
    try:
        image = Image.open(image_path)
        cropped_image = image.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))
        cropped_image.save(os.path.join(output_folder, os.path.basename(image_path)))
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_image_info(info_file, logos_folder, output_folder):
    with open(info_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 7:
            image_name, brand, _, x1, y1, x2, y2 = parts
            coordinates = (int(x1), int(y1), int(x2), int(y2))
            
            # Create a subfolder for each brand if it doesn't exist
            brand_folder = os.path.join(output_folder, brand)
            if not os.path.exists(brand_folder):
                os.makedirs(brand_folder)

            # Load the image and crop it
            image_path = os.path.join(logos_folder, image_name)
            crop_and_save(image_path, coordinates, brand_folder)



def remove_broken_images(root_folder):
    for brand_folder in os.listdir(root_folder):
        brand_path = os.path.join(root_folder, brand_folder)
        
        if os.path.isdir(brand_path):
            for image_name in os.listdir(brand_path):
                image_path = os.path.join(brand_path, image_name)
                
                try:
                    # Attempt to open the image
                    with Image.open(image_path) as img:
                        pass  # Do nothing if the image can be opened
                except UnidentifiedImageError:
                    # Remove the broken image
                    os.remove(image_path)
                    print(f"Removed broken image: {image_path}")

          
if __name__ == "__main__":
    info_file = "flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"  # Replace with the actual path to your info file
    logos_folder = "flickr_logos_dataset/flickr_logos_27_dataset_images"  # Replace with the actual path to your Logos folder
    output_folder = "flickr_logos_dataset/Cropped_Logos"  # Replace with the desired output folder

    process_image_info(info_file, logos_folder, output_folder)
    remove_broken_images(output_folder)
