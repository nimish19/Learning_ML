#Submission 17
from PIL import Image

# open image file 
img = Image.open("sample.jpg")
# conert image format into GreySale
img = img.convert('L')
#ROtateimage by 90 clockwise
img = img.rotate(-90)
#calculate width and height on image
w,h = img.size
#get center Coordinates of image
hw,hh = w/2,h/2
#crop image by width = 160,height = 204
img = img.crop((hw-80,hh-102,hw+80,hh+102))
#Create thumbnail of 75,75
img.thumbnail((75,75))
#Save img in newSample.jpg
img.save('newSample.jpg')

print("newSample.jpg")