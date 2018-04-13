import cv2

img = cv2.imread('IMG/center_2018_04_11_20_43_19_231.jpg')

image = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[:, :, 1]
 
image = image.reshape(160, 320, 1)

image_width = 32
image_height = 16

image = cv2.resize(image, (image_width, image_height))

print(image.shape)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
