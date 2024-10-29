import cv2
import supervision as sv
import os
from polygon_test import PolygonTest

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Loitering:

    def __init__(self, model, line_zones):

        self.frame = None
        self.model = model
        self.line_zones = line_zones
        self.detections =None
        self.count = None
        self.wait_time = 5
        self.Polygon_Test = PolygonTest()

    def predict(self, frame):
        try:
            self.frame = frame.get()
            box_annotator = sv.BoxAnnotator(
                thickness=2,
            )
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=0)

            for result in self.model.track(source=self.frame, classes=0, verbose=False, persist=True):

                self.frame = result.orig_img
                self.detections = sv.Detections.from_ultralytics(result)

                labels = [
                    f"{tracker_id} {self.model.names[class_id]}"
                    for box, mask, confidence, class_id, tracker_id, class_name
                    in self.detections
                ]
                box_annotator.annotate(
                    scene=self.frame,
                    detections=self.detections,

                )
                label_annotator.annotate(scene=self.frame, detections=self.detections, labels=labels)
                # return frame, detections
                self.polygon_test()
                return self.frame
        except Exception as er:
            print(er)

    def polygon_test(self):
        try:
            intersect, self.count = self.Polygon_Test.point_polygon_test(self.detections, self.line_zones)

            if intersect:
                self.plots()
            else:
                pass
        except Exception as er:
            print(er)

    def plots(self):
        try:
            cv2.putText(
                img=self.frame,
                text=f"Warning {self.count}",  # Shortened text
                org=(150, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Changed font style
                fontScale=1,  # Adjust font size
                color=(0, 0, 255),
                thickness=2  # Adjust thickness
            )
            cv2.polylines(self.frame, [self.line_zones], True, (0, 0, 255), 2)
        except Exception as er:
            print(er)