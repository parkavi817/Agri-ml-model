import gdown
import os

# Direct-download URL (correct format)
FILE_ID = "1DWY7nBMUiVttXNz0kdL83QyQKhGn1DTv"
url = f"https://drive.google.com/uc?id={FILE_ID}&confirm=t"

output = "plant_disease_model.h5"

# Download only if not already downloaded
if not os.path.exists(output):
    print("Downloading model.h5...")
    gdown.download(url, output, quiet=False)
    print("Download complete.")
else:
    print("plant_disease_model.h5 already exists.")
