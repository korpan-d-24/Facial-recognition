import cv2

# Haarcascades path
face_cascade = cv2.CascadeClassifier("C:\\Users\\Asus\\Desktop\\Facial-recognition\\haarcascade_frontalface_default.xml")

# Image path
img = cv2.imread("C:\\Users\\Asus\\Desktop\\Facial-recognition\\queen.jpg")

# Converts an image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecting object
faces = face_cascade.detectMultiScale(gray_img,
scaleFactor=1.1,
minNeighbors=5,
minSize=(30, 30))

# Rectangle
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Resizing images while maintaining the aspect ratio
resized = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

# Display the image, wait for the user to press a key
cv2.imshow("Gray", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
