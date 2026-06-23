import copy
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np


SCHEMA_VERSION = 1
DETECTOR_NAME = "SIFT"
ACTIVE_FILENAME = "active_sift.npz"


def utc_timestamp():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def file_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def map_stem_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def database_dir(map_stem, root="feature_dbs"):
    return os.path.join(root, map_stem)


def active_database_path(map_stem, root="feature_dbs"):
    return os.path.join(database_dir(map_stem, root), ACTIVE_FILENAME)


def build_metadata(map_profile, map_shape, view_size, lighting_profile="pre"):
    height, width = int(map_shape[0]), int(map_shape[1])
    now = utc_timestamp()
    return {
        "schema_version": SCHEMA_VERSION,
        "map_name": map_profile["name"],
        "map_filename": os.path.basename(map_profile["height_path"]),
        "map_stem": map_stem_from_path(map_profile["height_path"]),
        "map_dimensions": [width, height],
        "terrain_margin": map_profile["margin"],
        "map_scale": map_profile.get("map_scale"),
        "blur_sigma": map_profile.get("blur_sigma"),
        "detector_name": DETECTOR_NAME,
        "camera_fov_degrees": 45,
        "view_resolution": [int(view_size[0]), int(view_size[1])],
        "color_map_path": map_profile.get("color_path"),
        "pre_lighting_profile": lighting_profile,
        "created_at": now,
        "updated_at": now,
    }


def validate_metadata(metadata, map_profile, map_shape, view_size):
    expected = build_metadata(map_profile, map_shape, view_size)
    checks = [
        ("schema_version", metadata.get("schema_version"), SCHEMA_VERSION),
        ("map_stem", metadata.get("map_stem"), expected["map_stem"]),
        ("map_dimensions", metadata.get("map_dimensions"), expected["map_dimensions"]),
        ("detector_name", metadata.get("detector_name"), DETECTOR_NAME),
        ("camera_fov_degrees", metadata.get("camera_fov_degrees"), 45),
        ("terrain_margin", metadata.get("terrain_margin"), expected["terrain_margin"]),
    ]
    errors = []
    for name, got, want in checks:
        if got != want:
            errors.append(f"{name}: database has {got!r}, expected {want!r}")
    return errors


@dataclass
class ReferenceView:
    view_id: str
    pose: tuple
    keypoints: np.ndarray
    descriptors: np.ndarray
    created_at: str = field(default_factory=utc_timestamp)

    def mapped_count(self, mappings):
        return sum(1 for mapping in mappings if mapping["view_id"] == self.view_id)


class FeatureDatabase:
    def __init__(self, metadata, views=None, mappings=None):
        self.metadata = dict(metadata)
        self.views = list(views or [])
        self.mappings = list(mappings or [])

    @classmethod
    def empty(cls, map_profile, map_shape, view_size):
        return cls(build_metadata(map_profile, map_shape, view_size))

    def clone(self):
        return copy.deepcopy(self)

    def view_count(self):
        return len(self.views)

    def mapped_feature_count(self):
        return len(self.mappings)

    def descriptors_3d(self):
        if not self.mappings:
            return (
                np.empty((0, 128), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )
        desc = np.asarray([m["descriptor"] for m in self.mappings], dtype=np.float32)
        pts = np.asarray([m["world_point"] for m in self.mappings], dtype=np.float32)
        return desc, pts

    def add_reference_view(self, pose, keypoints, descriptors):
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape((-1, 2))
        descriptors = np.asarray(descriptors, dtype=np.float32).reshape((-1, 128))
        if len(keypoints) != len(descriptors):
            raise ValueError("keypoints and descriptors must have the same length")
        view = ReferenceView(
            view_id=f"view_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
            pose=tuple(float(v) for v in pose),
            keypoints=keypoints,
            descriptors=descriptors,
        )
        self.views.append(view)
        self.touch()
        return view

    def mapping_exists(self, view_id, keypoint_id):
        return any(
            m["view_id"] == view_id and int(m["keypoint_id"]) == int(keypoint_id)
            for m in self.mappings
        )

    def add_mapping(self, view_id, keypoint_id, world_point):
        view = self.get_view(view_id)
        if view is None:
            raise ValueError(f"unknown view_id {view_id}")
        keypoint_id = int(keypoint_id)
        if keypoint_id < 0 or keypoint_id >= len(view.keypoints):
            raise ValueError("keypoint_id out of range")
        if self.mapping_exists(view_id, keypoint_id):
            return None
        mapping = {
            "view_id": view_id,
            "keypoint_id": keypoint_id,
            "keypoint": tuple(float(v) for v in view.keypoints[keypoint_id]),
            "descriptor": np.asarray(view.descriptors[keypoint_id], dtype=np.float32),
            "world_point": tuple(float(v) for v in world_point),
            "created_at": utc_timestamp(),
        }
        self.mappings.append(mapping)
        self.touch()
        return mapping

    def remove_mapping(self, mapping):
        for index in range(len(self.mappings) - 1, -1, -1):
            item = self.mappings[index]
            if (item["view_id"] == mapping["view_id"]
                    and int(item["keypoint_id"]) == int(mapping["keypoint_id"])
                    and tuple(item["world_point"]) == tuple(mapping["world_point"])):
                del self.mappings[index]
                self.touch()
                return True
        return False

    def get_view(self, view_id):
        for view in self.views:
            if view.view_id == view_id:
                return view
        return None

    def mapped_keypoint_ids(self, view_id):
        return {
            int(m["keypoint_id"])
            for m in self.mappings
            if m["view_id"] == view_id
        }

    def touch(self):
        self.metadata["updated_at"] = utc_timestamp()

    def to_npz_arrays(self):
        view_ids = np.asarray([v.view_id for v in self.views], dtype="U64")
        view_poses = np.asarray([v.pose for v in self.views], dtype=np.float64).reshape((-1, 6))
        view_created_at = np.asarray([v.created_at for v in self.views], dtype="U32")
        ranges = []
        all_keypoints = []
        all_descriptors = []
        start = 0
        for view in self.views:
            count = len(view.keypoints)
            ranges.append((start, count))
            if count:
                all_keypoints.append(view.keypoints)
                all_descriptors.append(view.descriptors)
            start += count
        keypoints = (np.vstack(all_keypoints).astype(np.float32)
                     if all_keypoints else np.empty((0, 2), dtype=np.float32))
        descriptors = (np.vstack(all_descriptors).astype(np.float32)
                       if all_descriptors else np.empty((0, 128), dtype=np.float32))
        mapping_view_ids = np.asarray([m["view_id"] for m in self.mappings], dtype="U64")
        mapping_keypoint_ids = np.asarray([m["keypoint_id"] for m in self.mappings],
                                          dtype=np.int32)
        mapping_keypoints = np.asarray([m["keypoint"] for m in self.mappings],
                                       dtype=np.float32).reshape((-1, 2))
        mapping_descriptors = np.asarray([m["descriptor"] for m in self.mappings],
                                         dtype=np.float32).reshape((-1, 128))
        mapping_world = np.asarray([m["world_point"] for m in self.mappings],
                                   dtype=np.float32).reshape((-1, 3))
        mapping_created_at = np.asarray([m["created_at"] for m in self.mappings], dtype="U32")
        return {
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
            "view_ids": view_ids,
            "view_poses": view_poses,
            "view_created_at": view_created_at,
            "view_keypoint_ranges": np.asarray(ranges, dtype=np.int32).reshape((-1, 2)),
            "keypoints": keypoints,
            "descriptors": descriptors,
            "mapping_view_ids": mapping_view_ids,
            "mapping_keypoint_ids": mapping_keypoint_ids,
            "mapping_keypoints": mapping_keypoints,
            "mapping_descriptors": mapping_descriptors,
            "mapping_world_points": mapping_world,
            "mapping_created_at": mapping_created_at,
        }

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            view_ids = data["view_ids"]
            view_poses = data["view_poses"]
            view_created_at = data["view_created_at"]
            ranges = data["view_keypoint_ranges"]
            keypoints = data["keypoints"]
            descriptors = data["descriptors"]
            views = []
            for i, view_id in enumerate(view_ids):
                start, count = int(ranges[i][0]), int(ranges[i][1])
                views.append(ReferenceView(
                    view_id=str(view_id),
                    pose=tuple(float(v) for v in view_poses[i]),
                    keypoints=np.asarray(keypoints[start:start + count], dtype=np.float32),
                    descriptors=np.asarray(descriptors[start:start + count], dtype=np.float32),
                    created_at=str(view_created_at[i]),
                ))
            mappings = []
            for i, view_id in enumerate(data["mapping_view_ids"]):
                mappings.append({
                    "view_id": str(view_id),
                    "keypoint_id": int(data["mapping_keypoint_ids"][i]),
                    "keypoint": tuple(float(v) for v in data["mapping_keypoints"][i]),
                    "descriptor": np.asarray(data["mapping_descriptors"][i], dtype=np.float32),
                    "world_point": tuple(float(v) for v in data["mapping_world_points"][i]),
                    "created_at": str(data["mapping_created_at"][i]),
                })
            return cls(metadata, views, mappings)

    def save_active(self, map_stem, root="feature_dbs"):
        out_dir = database_dir(map_stem, root)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "backups"), exist_ok=True)
        active_path = active_database_path(map_stem, root)
        if os.path.exists(active_path):
            backup_name = f"active_sift_{file_timestamp()}.npz"
            shutil.copy2(active_path, os.path.join(out_dir, "backups", backup_name))
        tmp_path = active_path + ".tmp"
        np.savez_compressed(tmp_path, **self.to_npz_arrays())
        saved_tmp = tmp_path if os.path.exists(tmp_path) else tmp_path + ".npz"
        loaded = FeatureDatabase.load(saved_tmp)
        if loaded.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("saved database failed schema validation")
        os.replace(saved_tmp, active_path)
        return active_path

    def save_snapshot(self, map_stem, root="feature_dbs"):
        out_dir = database_dir(map_stem, root)
        snap_dir = os.path.join(out_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, f"sift_{file_timestamp()}.npz")
        np.savez_compressed(path, **self.to_npz_arrays())
        FeatureDatabase.load(path)
        return path


def load_active_database(map_profile, map_shape, view_size, root="feature_dbs"):
    map_stem = map_profile["name"]
    path = active_database_path(map_stem, root)
    if not os.path.exists(path):
        return None, path, [f"No feature database found for {map_stem}."]
    db = FeatureDatabase.load(path)
    errors = validate_metadata(db.metadata, map_profile, map_shape, view_size)
    return db, path, errors
