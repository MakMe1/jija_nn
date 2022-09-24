import cv2

bear = cv2.imread("bear.jpg")
h, w, d = bear.shape
scale = 8

print(f"{h=} {w=} {d=}")
resized = cv2.resize(bear, ((int)(w / scale), (int)(h / scale)), interpolation=cv2.INTER_AREA)
resized = cv2.resize(bear, (w, h), interpolation = cv2.INTER_AREA)

cv2.imshow("bear", bear)
cv2.imshow("resized", resized)
cv2.waitKey(0)