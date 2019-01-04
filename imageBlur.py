#Image Bluringfrom PIL import Image
#take an Image and take a filter array
'''
from PIL import Image
import numpy as np
img = Image.open('sample1.png')
img = img.convert('L')
'''
import numpy as np
kernel = 1/9*np.array([[1,1,1],[1,1,1],[1,1,1]])

from skimage import io, viewer
img1 = io.imread('sample1.png', as_gray=True)
#multiply pixel values of image with filter values while sliding the filter matrix and then add them all into a single values
# convolution output
output = np.zeros_like(img1)
# Add zero padding to the input image
image_padded = np.pad(img1,((1,1),(1,1)),'constant')
# Loop over every pixel of the image
for x in range(img1.shape[0]):     
    for y in range(img1.shape[1]):
        output[x,y]=(kernel*image_padded[x:x+3,y:y+3]).sum()        
import matplotlib.pyplot as plt
plt.imshow(output)
plt.axis('off')
plt.show()
