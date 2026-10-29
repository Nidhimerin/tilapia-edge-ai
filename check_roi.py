import cv2, sys
v = sys.argv[1]
cap = cv2.VideoCapture(v)
cap.set(cv2.CAP_PROP_POS_MSEC, 100_000)
ok, f = cap.read()
print(v, f.shape)
cv2.rectangle(f, (180, 250), (1180, 650), (0, 255, 0), 2)     # ROI
cv2.rectangle(f, (180, 260), (480, 650), (0, 0, 255), 2)      # near
cv2.imwrite(sys.argv[1].split("/")[-1] + "_roi.jpg", f)