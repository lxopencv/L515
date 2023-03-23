import cv2

# Define the dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Generate the marker
marker = cv2.aruco.drawMarker(dictionary, 23, 200)

# Resize the marker to 100mm x 100mm
marker = cv2.resize(marker, (100, 100))

# Save the marker as qr.png
cv2.imwrite('qr.png', marker)
