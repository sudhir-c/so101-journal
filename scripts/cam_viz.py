import cv2
cap = cv2.VideoCapture(0)   # your camera's Id
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # native max = widest FOV on most webcams
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_ZOOM, 0)             # minimize digital zoom (often unsupported)
print("zoom:", cap.get(cv2.CAP_PROP_ZOOM), "res:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()