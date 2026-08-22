from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class CameraCalibration:
    """Preparation-only interface; it never claims metric validity by default."""

    camera_height: Optional[float] = None
    camera_pitch: Optional[float] = None
    intrinsics: Optional[Sequence[float]] = None
    extrinsics: Optional[Sequence[float]] = None
    ground_plane: Optional[Sequence[float]] = None
    homography: Optional[Sequence[float]] = None
    status: str = "NOT CALIBRATED"
    notes: list[str] = field(default_factory=lambda: [
        "No camera calibration has been performed.",
        "Metric distance and real-world velocity are disabled.",
    ])

    @property
    def valid(self) -> bool:
        return self.status == "VALID"

    def image_to_ground(self, x: float, y: float):
        if not self.valid or self.homography is None:
            raise RuntimeError("Calibration status is not VALID; image-space coordinates only")
        raise NotImplementedError("Ground-plane projection requires a validated calibration implementation")

    def distance_label(self) -> str:
        return "METRIC" if self.valid else "IMAGE-SPACE"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "valid": self.valid,
            "camera_height": self.camera_height,
            "camera_pitch": self.camera_pitch,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "ground_plane": self.ground_plane,
            "homography": self.homography,
            "distance_unit": self.distance_label(),
            "notes": list(self.notes),
        }
