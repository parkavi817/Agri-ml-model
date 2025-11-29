import gdown

url = "https://drive.google.com/file/d/1DWY7nBMUiVttXNz0kdL83QyQKhGn1DTv/view?usp=sharing"
output = "model.h5"
gdown.download(url, output, quiet=False)