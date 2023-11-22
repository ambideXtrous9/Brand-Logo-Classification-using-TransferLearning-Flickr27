import os
import requests

def download_images(image_urls, folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for url in image_urls:
        # Get the filename from the URL
        filename = os.path.join(folder_name, url.split("/")[-1])

        # Download the image
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
                print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {filename}")

if __name__ == "__main__":
    # Read image URLs from the text file
    with open("flickr_logos_27_dataset/flickr_logos_27_dataset_distractor_set_urls.txt", "r") as file:
        image_urls = [line.strip() for line in file]

    # Specify the folder name
    folder_name = "flickr_logos_27_dataset/Distractor"

    # Download and store the images
    download_images(image_urls, folder_name)
