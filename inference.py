"""
interfaces.py — Type contracts (Protocols) for the AI Facial Expression Recognition project.

This file defines the *structural types* of the callables/objects that app.py
depends on. It does not import the implementations — only describes their shapes.

Nothing at runtime imports this file automatically. Use it in type annotations:

    from interfaces import FaceDetector, FaceEncoder, EmotionModel, CameraManagerAPI

    def process(detector: FaceDetector, encoder: FaceEncoder, model: EmotionModel):
        ...

This is purely for static type checking with pyright / mypy / IDE autocomplete.
"""
from __future__ import annotations

from typing import (
    Protocol, runtime_checkable, Optional, Tuple, List, Dict, Any,
    Iterator, Mapping, Sequence,
)
import numpy as np


# ---------------------------------------------------------------------------
# Utils module contracts
# ---------------------------------------------------------------------------

BBox = Tuple[int, int, int, int]   # (x, y, w, h)
Encoding = np.ndarray              # shape (128,), float64
FaceImage = np.ndarray             # BGR image from OpenCV
PreprocessedBatch = np.ndarray     # shape (1, H, W, 3), float32 in [0, 1]


@runtime_checkable
class FaceDetector(Protocol):
    """Matches `utils.detect_faces`."""
    def __call__(
        self,
        image: FaceImage,
        min_size: Tuple[int, int] = (40, 40),
    ) -> List[BBox]: ...


@runtime_checkable
class FacePreprocessor(Protocol):
    """Matches `utils.preprocess_face`."""
    def __call__(
        self,
        face_img: FaceImage,
        target_size: Tuple[int, int] = (224, 224),
    ) -> PreprocessedBatch: ...


# ---------------------------------------------------------------------------
# Face recognition module contracts
# ---------------------------------------------------------------------------

@runtime_checkable
class FaceEncoder(Protocol):
    """Matches `face_recognition_module.encode_face`."""
    def __call__(self, image_array: FaceImage) -> Optional[Encoding]: ...


@runtime_checkable
class EncodingToBytes(Protocol):
    def __call__(self, encoding: Optional[Encoding]) -> Optional[bytes]: ...


@runtime_checkable
class BytesToEncoding(Protocol):
    def __call__(self, data: Optional[bytes]) -> Optional[Encoding]: ...


@runtime_checkable
class FaceMatcher(Protocol):
    """
    Matches `face_recognition_module.find_matching_person`.

    Returns (person_name or None, confidence_percentage 0..100).
    """
    def __call__(
        self,
        incoming_encoding: Encoding,
        stored_encodings_dict: Mapping[str, Sequence[Encoding]],
        tolerance: float = 0.6,
    ) -> Tuple[Optional[str], float]: ...


@runtime_checkable
class FaceComparator(Protocol):
    """
    Matches `face_recognition_module.compare_faces`.

    Returns (is_match, distance).
    """
    def __call__(
        self,
        encoding1: Encoding,
        encoding2: Encoding,
        tolerance: float = 0.6,
    ) -> Tuple[bool, float]: ...


# ---------------------------------------------------------------------------
# Emotion model contract
# ---------------------------------------------------------------------------

@runtime_checkable
class EmotionModel(Protocol):
    """
    Subset of Keras `Model` we actually use.

    Loaded via `tensorflow.keras.models.load_model` or `keras.models.load_model`.
    """
    input_shape: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
    output_shape: Tuple[Optional[int], ...]

    def predict(
        self,
        x: Any,
        batch_size: Optional[int] = ...,
        verbose: int = ...,
        **kwargs: Any,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Camera manager contract
# ---------------------------------------------------------------------------

@runtime_checkable
class CameraManagerAPI(Protocol):
    """
    Matches `camera_manager.camera_manager` (the singleton instance).

    Both `get_camera` and `release` are used from app.py.
    """
    def get_camera(self) -> Tuple[Any, int]:
        """
        Returns (cv2.VideoCapture, camera_index).
        `cap` may be None if no camera is available.
        """
        ...

    def release(self) -> None: ...


# ---------------------------------------------------------------------------
# Database module contracts (structural, not exhaustive)
# ---------------------------------------------------------------------------

PersonRow = Dict[str, Any]           # {id, username, name, email, phone, created_at}
EncodingBytes = bytes
PersonId = int


@runtime_checkable
class PersonStore(Protocol):
    """Matches `database` functions used by app.py."""
    def find_person_by_username(self, username: str) -> Optional[PersonRow]: ...
    def register_person(
        self, username: str, name: str, email: str = "", phone: str = ""
    ) -> PersonRow: ...
    def load_all_people(self) -> List[PersonRow]: ...
    def delete_person(self, username: str) -> None: ...


@runtime_checkable
class DetectionLogger(Protocol):
    def log_face_detection(
        self,
        person_name: str,
        emotion: str,
        person_id: Optional[PersonId] = None,
    ) -> None: ...

    def get_face_detection_logs(self, limit: int = 100) -> List[Dict[str, Any]]: ...
    def get_person_logs(self, person_name: str, limit: int = 50) -> List[Dict[str, Any]]: ...
    def get_logs_by_date(self, from_date: str, to_date: str) -> List[Dict[str, Any]]: ...
    def get_recent_logs(self, limit: int = 5) -> List[Dict[str, Any]]: ...


@runtime_checkable
class FaceEncodingStore(Protocol):
    def store_face_encoding(
        self, person_id: PersonId, encoding_bytes: EncodingBytes
    ) -> int: ...

    def get_face_encodings_for_person(
        self, person_id: PersonId
    ) -> List[EncodingBytes]: ...

    def get_all_face_encodings(self) -> Dict[str, List[EncodingBytes]]: ...

    def clear_face_encodings_for_person(self, person_id: PersonId) -> None: ...


@runtime_checkable
class AuthStore(Protocol):
    def create_user(
        self, username: str, password: str, is_admin: int = 0
    ) -> bool: ...

    def verify_user_password(
        self, username: str, password: str
    ) -> Tuple[bool, bool]: ...

    def update_user_password(self, username: str, new_password: str) -> bool: ...
    def is_user_admin(self, username: str) -> bool: ...
    def user_exists_in_users(self, username: str) -> bool: ...


@runtime_checkable
class StatsProvider(Protocol):
    def get_total_people(self) -> int: ...
    def get_total_face_encodings(self) -> int: ...
    def get_total_logs(self) -> int: ...
    def get_today_detection_count(self) -> int: ...
    def get_unique_people_detected(self) -> int: ...


# ---------------------------------------------------------------------------
# app.py internal state contracts (documentation only)
# ---------------------------------------------------------------------------

class FrameRenderer(Protocol):
    """
    Matches `app.render_frame`.

    Returns JPEG bytes for a single frame with boxes + labels drawn,
    or None on failure.
    """
    def __call__(
        self, frame: FaceImage, camera_index: Optional[int] = None
    ) -> Optional[bytes]: ...


class FrameGenerator(Protocol):
    """
    Matches `app.generate_frames`.

    Yields multipart/x-mixed-replace boundary chunks (each chunk is bytes
    containing headers + jpeg payload, ending in \\r\\n).
    """
    def __call__(self, display_name: str = "Unknown Person") -> Iterator[bytes]: ...


class FrameSupplier(Protocol):
    """Matches `app.get_one_frame`. Returns a single JPEG frame as bytes."""
    def __call__(self, display_name: str = "Unknown Person") -> Optional[bytes]: ...


class DetectionState(Protocol):
    """
    Snapshot of the module-level globals that /current_detection exposes.

    This is *informational only* — app.py holds these as module globals,
    not as an object. It documents the shape of the dict returned by the route.
    """
    person: str        # LAST_PERSON
    emotion: str       # LAST_EMOTION
    confidence: float  # LAST_CONFIDENCE (0..100)
    camera_running: bool  # CAMERA_RUNNING