import cv2
import numpy as np


def draw(image, windows_name):
    clone = image.copy()
    cv2.imshow(f"Draw Line - {windows_name}", clone)
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(clone, (x, y), 4, (0, 0, 255), thickness=-1)
            cv2.imshow(f"Draw Line - {windows_name}", clone)

    cv2.setMouseCallback(f"Draw Line - {windows_name}", click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == 13:  # 'Enter' key
            break

    cv2.destroyWindow(f"Draw Line - {windows_name}")

    if len(points) >= 2:
        if points[0] != points[-1]:
            points.append(points[0])
        line_coord = np.array(points, dtype=np.int32)
        print(f"Line coordinates saved for {windows_name}: {windows_name}")
        return line_coord, image

    else:
        print("Not enough points to form a line.")