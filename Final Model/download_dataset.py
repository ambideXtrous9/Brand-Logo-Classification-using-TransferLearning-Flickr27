import os
import requests
import tarfile

def download_and_extract(url, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the filename from the URL
    filename = os.path.join(destination_folder, url.split("/")[-1])

    # Download the file
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    # Extract the contents of the tar.gz file
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(destination_folder)

    # Remove the downloaded compressed file
    os.remove(filename)

if __name__ == "__main__":
    dataset_url = "http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz"
    
    destination_folder = "flickr_logos_dataset"

    download_and_extract(dataset_url, destination_folder)
    
    fname = 'flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz'
    
    with tarfile.open(fname, 'r:gz') as tar:
        tar.extractall(destination_folder)

    os.remove(fname)
    
    
    
    
