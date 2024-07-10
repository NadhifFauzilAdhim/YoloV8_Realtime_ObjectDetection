import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(7)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    object_counter = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        object_counter.clear()
        for _, confidence, class_id, _ in detections:
            class_name = model.model.names[class_id]
            if class_name in object_counter:
                object_counter[class_name] += 1
            else:
                object_counter[class_name] = 1

        # Display object counts
        y_offset = 30
        for class_name, count in object_counter.items():
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
            y_offset += 30
        
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
