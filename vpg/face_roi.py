import numpy as np


def clamp_roi(x, y, w, h, frame_width, frame_height):
    x1 = max(0, min(frame_width - 1, int(round(x))))
    y1 = max(0, min(frame_height - 1, int(round(y))))
    x2 = max(0, min(frame_width, int(round(x + w))))
    y2 = max(0, min(frame_height, int(round(y + h))))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def get_forehead_roi(face_landmarks, frame_width, frame_height):
    landmarks = getattr(face_landmarks, "landmark", face_landmarks)
    points = np.array([
        [landmark.x * frame_width, landmark.y * frame_height]
        for landmark in landmarks
    ])

    face_x_min, face_y_min = np.min(points, axis=0)
    face_x_max, face_y_max = np.max(points, axis=0)
    face_w = face_x_max - face_x_min
    face_h = face_y_max - face_y_min

    brow_points = points[[70, 63, 105, 66, 107, 336, 296, 334, 293, 300]]
    brow_center = np.mean(brow_points, axis=0)
    brow_y = float(np.mean(brow_points[:, 1]))

    forehead_top = points[10]
    forehead_span = max(8.0, brow_y - forehead_top[1])

    roi_w = 0.36 * face_w
    roi_h = min(0.36 * face_h, 1 * forehead_span)
    roi_x = brow_center[0] - roi_w / 2 + 0.03 * face_w
    roi_y = forehead_top[1] + 0.12 * forehead_span

    brow_margin = 0.22 * forehead_span
    max_roi_bottom = brow_y - brow_margin
    if roi_y + roi_h > max_roi_bottom:
        roi_h = max(6.0, max_roi_bottom - roi_y)

    face_box_top_padding = 0.18 * face_h
    roi = clamp_roi(roi_x, roi_y, roi_w, roi_h, frame_width, frame_height)
    face_box = clamp_roi(
        face_x_min,
        face_y_min - face_box_top_padding,
        face_w,
        face_h + face_box_top_padding,
        frame_width,
        frame_height,
    )
    return roi, face_box
