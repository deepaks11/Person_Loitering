import time
from shapely.geometry import Polygon
import datetime


class PolygonTest:

    def __init__(self, time_limit=5):
        self.detections = None
        self.zones = None
        self.polygon = None
        self.time_limit = time_limit  # Time limit for loitering
        self.object_id_list = []      # List of object IDs detected that are intersecting
        self.dtime = {}               # Time dictionary for loitering detection
        self.counter_time = {}        # Counter for tracking time spent in loitering area
        self.start_time = time.time()  # Initialize start time

    def point_polygon_test(self, detections, line_zones):
        try:
            self.detections = detections
            self.zones = line_zones
            self.polygon = Polygon(self.zones)  # Create polygon from the updated zones
            count = 0
            any_loitering = False  # Track if any object is loitering

            # Check if a minute has passed to clear the object_id_list
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 60:
                self.object_id_list.clear()
                self.start_time = time.time()  # Reset start time after clearing the list

            if self.detections is not None and self.detections.tracker_id is not None:
                for i in range(len(self.detections.xyxy)):  # Loop through all detected objects
                    p_x1, p_y1, p_x2, p_y2 = self.detections.xyxy[i].astype(int)
                    person_coord = [(p_x1, p_y1), (p_x1, p_y2), (p_x2, p_y1), (p_x2, p_y2)]
                    obj_id = self.detections.tracker_id[i]

                    # Create a rectangle polygon for the detected bounding box
                    rect_polygon = Polygon(person_coord)

                    # Check if the bounding box intersects with the polygon
                    if self.polygon.intersects(rect_polygon):
                        count += 1

                        # Loitering logic - track time spent in the area
                        if obj_id not in self.object_id_list:
                            # Only add to the list if it intersects
                            self.object_id_list.append(obj_id)
                            self.dtime[obj_id] = datetime.datetime.now()  # Set the entry time
                            self.counter_time[obj_id] = 0  # Reset counter time
                        else:
                            # If already in the list, check how long it has been in the area
                            curr_time = datetime.datetime.now()
                            old_time = self.dtime[obj_id]
                            time_diff = curr_time - old_time
                            sec = time_diff.total_seconds()

                            # Update the time for loitering detection
                            self.dtime[obj_id] = curr_time
                            self.counter_time[obj_id] += sec

                            # Check if the object has exceeded the time limit in the loitering zone
                            if self.counter_time[obj_id] >= self.time_limit:
                                any_loitering = True

                return any_loitering, count  # Return whether any object is loitering and count of objects inside
            else:
                return any_loitering, count

        except Exception as ex:
            print(f"Error: {ex}")
            return False, 0  # Return False and 0 if an exception occurs
