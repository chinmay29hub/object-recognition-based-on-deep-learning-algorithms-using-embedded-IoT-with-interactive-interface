import cv2

# Read image
img = cv2.imread('3.jpg')

# Get x and y coordinates
height, width = img.shape[:2]
x_coordinates = range(width)
y_coordinates = range(height)

# Print the x and y coordinates
print("X coordinates: {}".format(x_coordinates))
print("Y coordinates: {}".format(y_coordinates))
