
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return abs(x2 - x1)

def get_foot_position(bbox):
    x_center, _ = get_center_of_bbox(bbox)
    return (x_center, bbox[3])

def draw_tracks(frame, tracks):
    import cv2
    for tid, bbox in tracks:
        x1, y1, x2, y2, _ = bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

def save_video(frames, path, fps=30):
    import cv2
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    for f in frames:
        out.write(f)
    out.release()
