
import cv2
import yaml
from detector import detect_players
from tracker import Tracker
from reid_model import ReIDMatcher
from utils import draw_tracks, save_video

def main():
    # Load config
    cfg = yaml.safe_load(open("config.yaml", "r"))
    video_in = cfg["video"]["input_path"]
    video_out = cfg["video"]["output_path"]

    cap = cv2.VideoCapture(video_in)
    frames = []

    matcher = ReIDMatcher(similarity_threshold=cfg["thresholds"]["reid_similarity"])
    tracker = Tracker(matcher)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_players(frame, conf_th=cfg["thresholds"]["detection_confidence"])
        tracked_objects = tracker.update(frame, detections)
        out_frame = draw_tracks(frame, tracked_objects)
        frames.append(out_frame)
        cv2.imshow("Re-ID Tracking", out_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_video(frames, video_out)

if __name__ == "__main__":
    main()
