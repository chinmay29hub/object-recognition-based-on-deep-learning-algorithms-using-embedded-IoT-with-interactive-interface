import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print("Mouse position: x={}, y={}".format(x, y))

# Load an image
img = cv2.imread("cars.jpg")

# Create a window to display the image
cv2.namedWindow("Image")

# Set the mouse callback function for the window
cv2.setMouseCallback("Image", mouse_callback)

# Display the image
cv2.imshow("Image", img)

# Wait for a key press
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
