import cv2

image = cv2.imread("car.jpg")
car_cascade = cv2.CascadeClassifier("car.xml")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

for car in cars:
    (x, y, w, h) = car
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
