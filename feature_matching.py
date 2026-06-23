import cv2
import numpy as np


def make_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "cv2.SIFT_create not available - install opencv-contrib-python "
            "or a recent opencv-python build with SIFT support."
        )
    return cv2.SIFT_create()


def make_flann():
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def detect_sift_features(bgr_image):
    sift = make_sift()
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)
    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    return points, np.asarray(descriptors, dtype=np.float32)


def build_matcher(database):
    descriptors, points3d = database.descriptors_3d()
    if len(descriptors) == 0:
        return None, descriptors, points3d
    matcher = make_flann()
    matcher.add([descriptors])
    matcher.train()
    return matcher, descriptors, points3d


def match_query_to_database(database, query_bgr, ratio=0.7):
    query_keypoints, query_descriptors = detect_sift_features(query_bgr)
    matcher, mapped_descriptors, mapped_points = build_matcher(database)
    metrics = {
        "query_keypoint_count": int(len(query_keypoints)),
        "good_descriptor_match_count": 0,
    }
    if matcher is None or len(mapped_descriptors) < 2:
        return [], query_keypoints, metrics, "database has fewer than two mapped descriptors"
    if len(query_descriptors) < 2:
        return [], query_keypoints, metrics, "query has fewer than two descriptors"

    knn = matcher.knnMatch(query_descriptors, k=2)
    candidates = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            candidates.append(first)

    candidates.sort(key=lambda m: float(m.distance))
    used_query = set()
    used_train = set()
    correspondences = []
    for match in candidates:
        if match.queryIdx in used_query or match.trainIdx in used_train:
            continue
        used_query.add(match.queryIdx)
        used_train.add(match.trainIdx)
        qx, qy = query_keypoints[match.queryIdx]
        wx, wy, wz = mapped_points[match.trainIdx]
        correspondences.append((
            (float(qx), float(qy)),
            (float(wx), float(wy), float(wz)),
        ))

    metrics["good_descriptor_match_count"] = int(len(correspondences))
    return correspondences, query_keypoints, metrics, None
