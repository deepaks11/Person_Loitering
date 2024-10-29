import os
import cv2
import threading
import queue
from ultralytics import YOLO
from multiprocessing.pool import ThreadPool
from loitering_detection import Loitering
from draw_line import draw
queue_name = queue.Queue()
pool = ThreadPool(processes=1)
root = os.getcwd()


class VideoCapture:

    def __init__(self, rtsp_url):

        self.cap = cv2.VideoCapture(rtsp_url)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True  # Set the thread as daemon to stop it when the main program exits
        t.start()

    def _reader(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class PlayVideo:

    def __init__(self, source, window_name, q):
        self.yolo = None
        self.cap_line = None
        self.model = YOLO("yolov8l.pt")
        self.detections = None
        self.image = None
        self.obj = None
        self.frame = None
        self.resized_img = None
        self.source = source
        self.window_name = window_name
        self.q_img = q
        self.line = False
        self.line_coord = None

    def vdo_cap(self):

        try:

            if self.source.startswith("rtsp"):
                self.cap_line = VideoCapture(self.source)
            else:
                self.cap_line = cv2.VideoCapture(self.source)

            while True:
                # if you want to read any different video file format just add below
                if self.source.endswith((".mp4", ".avi")):
                    ret, self.image = self.cap_line.read()

                else:
                    self.image = self.cap_line.read()

                self.image = cv2.resize(self.image, (1080, 720))
                if not self.line:
                    self.line_coord, self.image = draw(self.image, self.window_name)
                    self.line = True
                    self.yolo = Loitering(self.model, self.line_coord)

                if self.line_coord is not None:

                    cv2.polylines(self.image, [self.line_coord], True, (0, 220, 0), 2)

                self.q_img.put(self.image)
                obj = pool.apply_async(self.yolo.predict, (self.q_img, ))
                frame = obj.get()
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) == ord('q'):  # Exit if 'q' is pressed
                    break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    urls = [

        # {"name": "rtsp2", "url": r"2023-07-28_16-00-17.mp4"},
        {"name": "rtsp3", "url": r"rtsp://admin:Admin123$@10.11.25.65:554/stream1"},
        # {"name": "rtsp4", "url": r"rtsp://admin:Admin123$@10.11.25.57:554/stream1"},
        # {"name": "rtsp5", "url": r"rtsp://admin:Admin123$@10.11.25.58:554/stream1"},
        # {"name": "rtsp6", "url": r"rtsp://admin:Admin123$@10.11.25.51:554/stream1"},
        # {"name": "rtsp7", "url": r"rtsp://admin:Admin123$@10.11.25.52:554/stream1"},
        # {"name": "rtsp8", "url": r"rtsp://admin:Admin123$@10.11.25.53:554/stream1"},
        # {"name": "rtsp9", "url": r"rtsp://admin:Admin123$@10.11.25.54:554/stream1"},
        # {"name": "rtsp10", "url": r"rtsp://admin:Admin123$@10.11.25.55:554/stream1"},
        # {"name": "rtsp11", "url": r"rtsp://admin:Admin123$@10.11.25.59:554/stream1"},
        {"name": "rtsp12", "url": r"rtsp://admin:Admin123$@10.11.25.64:554/stream1"}
    ]
    queue_list = []
    threads = []
    for i in urls:
        url = i['url']
        name = i["name"]
        queue_name.name = name
        queue_list.append(queue_name)
        td = threading.Thread(
            target=PlayVideo(url, name, queue_name).vdo_cap)
        td.start()
        threads.append(td)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()