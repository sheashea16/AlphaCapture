import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Webcam opened. Press SPACE to capture, ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam - Press SPACE to capture", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        print("Exiting without capture.")
        break
    elif key == 32:  # SPACE
        cv2.imwrite("reference.jpg", frame)
        print("Saved reference.jpg")
        break

cap.release()
cv2.destroyAllWindows()
