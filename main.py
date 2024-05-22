#main.py

import requests
import zipfile
import io

# Download the file
url = "https://www.dropbox.com/s/nilt43hyl1dx82k/dataset.zip?dl=1"  # Used dl=1 to force the browser to download
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the contents
zip_file.extractall("data")
zip_file.close()