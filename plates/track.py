# ─── Temporal tracking (PlateHistory, VehicleTrack, SceneTracker) ──────────
import math
import os

import cv2
import numpy as np

from plates.common import _overlaps, covered_fraction
from plates.kalman import BoxFilter


def _clamp_to_box(zone: tuple, box: tuple, frac: float = 0.10) -> tuple:
    """Move *zone* inside *box* expanded by *frac* of the box's size.

    The plate belongs to the vehicle, so however a prediction is produced it may
    not leave the vehicle's neighbourhood: that is what stops a long coast from
    sliding onto roadside objects while the vehicle box stays put.

    The zone is translated rather than intersected. Intersecting looks equivalent
    until the prediction ends up entirely outside the box, where it yields an
    empty rectangle and the caller is left with the very zone that had to be
    corrected — the plate then appears to teleport when the zone finally comes
    from somewhere else. Translating keeps the zone's size and always lands it on
    the vehicle.
    """
    bx1, by1, bx2, by2 = box
    px = (bx2 - bx1) * frac
    py = (by2 - by1) * frac
    lx1, ly1 = bx1 - px, by1 - py
    lx2, ly2 = bx2 + px, by2 + py
    x1, y1, x2, y2 = zone
    w, h = x2 - x1, y2 - y1
    if w >= lx2 - lx1:                 # wider than the limit: centre it
        x1 = (lx1 + lx2 - w) * 0.5
    else:
        x1 = min(max(x1, lx1), lx2 - w)
    if h >= ly2 - ly1:
        y1 = (ly1 + ly2 - h) * 0.5
    else:
        y1 = min(max(y1, ly1), ly2 - h)
    return (int(x1), int(y1), int(x1 + w), int(y1 + h))


class PlateHistory:
    """
    Rolling history of confirmed plate detections for one tracked vehicle.
    Provides velocity-based position prediction for gap frames.
    """

    def __init__(self, max_history: int = 15):
        self._data: list = []          # (frame_idx, cx, cy, w, h, conf)
        self.max_history = max_history
        self.miss_count  = 0           # consecutive frames without a detection
        self._lk_pts: np.ndarray  = None   # shape (N,1,2) float32, full-frame px
        self._lk_gray: np.ndarray = None   # grayscale frame where _lk_pts were last set

    @property
    def has_history(self) -> bool:
        return len(self._data) > 0

    def record(self, frame_idx: int, x1, y1, x2, y2, conf: float):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w  = float(x2 - x1)
        h  = float(y2 - y1)
        self._data.append((frame_idx, cx, cy, w, h, conf))
        if len(self._data) > self.max_history:
            self._data.pop(0)
        self.miss_count = 0

    def predict_rect(self, frame_idx: int, max_expand: int = 20,
                     max_px_per_frame: float = 40.0):
        """
        Return (x1, y1, x2, y2) predicted at frame_idx.

        Velocity is computed as an exponentially-weighted average of
        per-frame deltas so recent movement dominates.  The box grows by
        2 px per missed frame to account for growing positional uncertainty,
        capped at max_expand pixels per side so a large gap_frames setting
        doesn't balloon the blur region across the frame.

        A displacement sanity cap rejects implausible extrapolations: when a
        plate is lost for several frames the velocity model drifts onto trees,
        bins, road furniture etc. If the extrapolated centre would have moved
        more than max_px_per_frame px per missed frame, we return None so the
        caller blinks instead of blurring the wrong spot.

        Returns None if no history exists.
        """
        if not self._data:
            return None

        _, cx_last, cy_last, w_last, h_last, _ = self._data[-1]
        f_last = self._data[-1][0]

        if len(self._data) >= 2:
            vxs, vys, weights = [], [], []
            for i in range(1, len(self._data)):
                f0, cx0, cy0 = self._data[i-1][0], self._data[i-1][1], self._data[i-1][2]
                f1, cx1, cy1 = self._data[i][0],   self._data[i][1],   self._data[i][2]
                dt = f1 - f0
                if dt > 0:
                    vxs.append((cx1 - cx0) / dt)
                    vys.append((cy1 - cy0) / dt)
                    weights.append(2.0 ** i)        # exponential: recent frames dominate
            if vxs:
                tw = sum(weights)
                vx = sum(v * w for v, w in zip(vxs, weights)) / tw
                vy = sum(v * w for v, w in zip(vys, weights)) / tw
                dt = frame_idx - f_last
                cx_last = cx_last + vx * dt
                cy_last = cy_last + vy * dt

        # Displacement sanity cap: reject drift beyond plausible motion
        gap = max(1, frame_idx - f_last)
        if abs(cx_last - self._data[-1][1]) > max_px_per_frame * gap or \
           abs(cy_last - self._data[-1][2]) > max_px_per_frame * gap:
            return None

        # Grow 2 px per missed frame, but never more than max_expand px per side
        expand = min(self.miss_count * 2, max_expand)
        x1 = int(cx_last - w_last * 0.5 - expand)
        y1 = int(cy_last - h_last * 0.5 - expand)
        x2 = int(cx_last + w_last * 0.5 + expand)
        y2 = int(cy_last + h_last * 0.5 + expand)
        return (x1, y1, x2, y2)

    def refresh_lk(self, gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        (Re-)initialise Lucas-Kanade tracking from a freshly-confirmed plate bbox.

        Extracts corner features from the plate crop and stores them alongside
        the grayscale frame so predict_rect_lk() can propagate them forward.
        Falls back silently when the crop has too few trackable corners.
        """
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return
        pts = cv2.goodFeaturesToTrack(
            crop, maxCorners=16, qualityLevel=0.1, minDistance=3, blockSize=5,
        )
        if pts is None or len(pts) < 3:
            return
        pts[:, 0, 0] += x1   # offset crop-space → full-frame space
        pts[:, 0, 1] += y1
        self._lk_pts  = pts
        self._lk_gray = gray.copy()

    def predict_rect_lk(
        self, gray: np.ndarray, max_expand: int = 20,
        max_px_per_frame: float = 40.0,
    ) -> "tuple[int,int,int,int] | None":
        """
        Propagate stored corner points to *gray* via Lucas-Kanade sparse optical
        flow, then derive a bounding box from where they landed.

        Updates the stored points and frame so chained gap-fill calls each build
        on the latest tracked position.  Returns None when too few points survive
        (caller should fall back to velocity prediction).

        A displacement sanity cap also returns None when the tracked points jump
        more than max_px_per_frame px in one frame — that means the points have
        latched onto background texture (trees, roadside objects) and the derived
        box would blur the wrong location.
        """
        if self._lk_pts is None or self._lk_gray is None or not self._data:
            return None

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._lk_gray, gray, self._lk_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if new_pts is None or status is None:
            return None

        good = new_pts[status.ravel() == 1].reshape(-1, 2)   # (M, 2)
        if len(good) < 3:
            return None

        # Sanity: if the tracked points displaced more than max_px_per_frame,
        # they latched onto background — reject this prediction entirely.
        prev_mean = self._lk_pts.reshape(-1, 2).mean(axis=0)
        new_mean  = good.mean(axis=0)
        if np.hypot(*(new_mean - prev_mean)) > max_px_per_frame:
            return None

        _, _, _, w_last, h_last, _ = self._data[-1]
        cx = float(new_mean[0])
        cy = float(new_mean[1])

        expand = min(self.miss_count * 2, max_expand)
        x1 = int(cx - w_last * 0.5 - expand)
        y1 = int(cy - h_last * 0.5 - expand)
        x2 = int(cx + w_last * 0.5 + expand)
        y2 = int(cy + h_last * 0.5 + expand)

        # Update for the next gap frame — LK expects (N,1,2)
        self._lk_pts  = good.reshape(-1, 1, 2).astype(np.float32)
        self._lk_gray = gray.copy()
        return (x1, y1, x2, y2)

class VehicleTrack:
    """One tracked vehicle with its associated plate history."""
    _id_counter = 0

    def __init__(self, box: tuple, frame_idx: int, history_frames: int = 15,
                 cls: int = 0):
        VehicleTrack._id_counter += 1
        self.id          = VehicleTrack._id_counter
        self.box         = box           # (x1, y1, x2, y2) full-res
        self.last_frame  = frame_idx
        self.miss_count  = 0
        self.frames_seen = 1             # frames vehicle was successfully detected
        self.cls         = int(cls)      # vehicle class id (motorcycle, car, ...)
        self.plate       = PlateHistory(max_history=history_frames)
        self.plate_anchor = None         # (ndx, ndy, nw, nh) — plate offset
                                         # normalised to the vehicle box. When the
                                         # plate is lost, we apply this offset to
                                         # the *current* vehicle box (which
                                         # ByteTrack keeps following), so the blur
                                         # physically stays glued to the vehicle
                                         # instead of drifting onto background.
        # EMA state for the plate box — smooths frame-to-frame jitter so the
        # blur doesn't flicker between detection positions.
        self.ema = None                  # (cx, cy, w, h)
        # Box history for velocity-based extrapolation while the vehicle
        # detector drops the motorcycle (light changes, inclination, occlusion).
        self.box_hist = []               # [(frame_idx, (x1,y1,x2,y2)), ...]
        self.box_vel  = None             # EMA velocity (dvx, dvy) px/frame
        # EMA state of the anchor zone itself — dimensional jumps of the blur
        # box between frames are smoothed so the zone glides, never snaps.
        self.zone_ema = None             # (cx, cy, w, h)
        # True right after a re-born track was seeded from a ghost zone: the
        # first emitted zone is the ghost itself (no EMA blend), so the blur
        # never jumps when the detector re-acquires the same motorcycle.
        self.zone_ghost_seeded = False
        # Plate-confirmed zone upgrade (moto anchor mode): when the plate model
        # reliably finds the plate inside this moto (conf >= moto_plate_conf
        # for moto_plate_promote_frames consecutive frames), the geometric
        # anchor zone is replaced by the EMA-smoothed plate detection — precise
        # at distance, while the geometric zone keeps covering the close-up
        # cases the model can't see. The streak decays after
        # moto_plate_promote_frames frames without a valid detection.
        self.moto_plate_streak = 0     # consecutive valid plate detections
        self.moto_plate_miss    = 0    # consecutive frames without one
        # Plate-track state (burst detection): velocity of the plate centre
        # and the frame index of the last confirmed detection. Used to HOLD
        # and extrapolate the promoted zone across detection gaps (angled
        # motos see the plate only in bursts) instead of snapping back to
        # the geometric anchor.
        self.plate_vel        = (0.0, 0.0)   # EMA (vx, vy) px/frame
        self.plate_prev       = None         # previous EMA centre (cx, cy)
        self.plate_last_seen  = -1           # frame_idx of last detection
        self.plate_zone_prev  = None         # centre of last EMITTED plate zone
        # Relative plate pose inside the vehicle box (rx, ry, rw, rh), learned
        # from geometrically-plausible detections. Used as a weak Kalman
        # measurement when the detector is silent: the blur stays where THIS
        # motorcycle's plate actually sits, not at a generic geometric guess.
        self.plate_rel        = None
        # Last box confirmed by a REAL detection (not Kalman-predicted by
        # ByteTrack). Close-up motos keep their anchor zone glued to this box
        # while the detector misses them — the Kalman extrapolation is
        # unreliable at close range and would drag the blur off the plate.
        self.last_detected_box   = box
        self.last_detected_frame = frame_idx
        # Last zone EMITTED for this track (frame_idx, cx, cy). The blur can
        # never travel more than SceneTracker.emit_max_disp px/frame from this
        # point: a plate confirmed 100% at (x, y) cannot legitimately show up
        # at (x+500, y+500) a few frames later — any zone that tries to
        # teleport is clamped back toward this safe position.
        self.last_emit = None      # (frame_idx, cx, cy)
        # Constant-velocity filter over the plate zone (zone_filter="kalman").
        # Replaces the ema/zone_ema cascade and the manual plate_vel
        # extrapolation: see plates/kalman.py. kf_frame records the frame the
        # filter was last advanced to, so predict() runs exactly once per frame
        # even when emission takes a different branch.
        self.kf         = None
        self.kf_frame   = frame_idx
        self.kf_rejects = 0      # consecutive detections rejected by the gate
        self.kf_pad     = 0.0    # uncertainty padding currently applied (px)
        # Last zone emitted WITHOUT clamping (a free emission: detection or
        # geometrically-consistent zone). Gap-fill extrapolations may never
        # wander more than SceneTracker.emit_max_total px from this confirmed
        # position — the blur stays in the plate's safe neighbourhood instead
        # of following a runaway vehicle box across the background.
        self.last_confirmed = None  # (frame_idx, cx, cy)

    def update_box(self, box: tuple, frame_idx: int, detected: bool = True):
        if detected:
            self.last_detected_box   = box
            self.last_detected_frame = frame_idx
        if self.box is not None and frame_idx > self.last_frame:
            vx = (box[0] + box[2]) - (self.box[0] + self.box[2])
            vy = (box[1] + box[3]) - (self.box[1] + self.box[3])
            v  = (vx * 0.5, vy * 0.5)
            if self.box_vel is None:
                self.box_vel = v
            else:
                self.box_vel = (0.6 * v[0] + 0.4 * self.box_vel[0],
                                0.6 * v[1] + 0.4 * self.box_vel[1])
        self.box_hist.append((frame_idx, box))
        if len(self.box_hist) > 8:
            self.box_hist.pop(0)
        self.box         = box
        self.last_frame  = frame_idx
        self.miss_count  = 0
        self.frames_seen += 1

    def predict_box(self, frame_idx: int, max_disp: float = 40.0):
        """Extrapolate the vehicle box from its EMA velocity.

        Used when the vehicle detector drops the motorcycle so the anchor zone
        keeps following the motion instead of freezing in place. Returns None
        when there is no velocity estimate or the implied motion is implausible
        (detector glitch → better to keep the last known box than to fly).
        """
        if self.box_vel is None:
            return None
        dt = max(1, frame_idx - self.last_frame)
        vx, vy = self.box_vel
        if abs(vx) > max_disp or abs(vy) > max_disp:
            return None
        x1, y1, x2, y2 = self.box
        cx, cy = (x1 + x2) * 0.5 + vx * dt, (y1 + y2) * 0.5 + vy * dt
        bw, bh = x2 - x1, y2 - y1
        return (int(cx - bw * 0.5), int(cy - bh * 0.5),
                int(cx + bw * 0.5), int(cy + bh * 0.5))

    def zone_ema_update(self, zone: tuple, alpha: float = 0.6):
        """EMA of the anchor zone (cx, cy, w, h) — smooths dimensional jumps."""
        cx = (zone[0] + zone[2]) * 0.5
        cy = (zone[1] + zone[3]) * 0.5
        w  = zone[2] - zone[0]
        h  = zone[3] - zone[1]
        if self.zone_ema is None:
            self.zone_ema = (cx, cy, w, h)
        else:
            ex, ey, ew, eh = self.zone_ema
            self.zone_ema = (alpha * cx + (1 - alpha) * ex,
                             alpha * cy + (1 - alpha) * ey,
                             alpha * w  + (1 - alpha) * ew,
                             alpha * h  + (1 - alpha) * eh)
        return self.zone_ema

    def ema_update(self, plate_box: tuple, alpha: float = 0.6):
        """Exponential moving average of the plate box (cx, cy, w, h)."""
        cx = (plate_box[0] + plate_box[2]) * 0.5
        cy = (plate_box[1] + plate_box[3]) * 0.5
        w  = plate_box[2] - plate_box[0]
        h  = plate_box[3] - plate_box[1]
        if self.ema is None:
            self.ema = (cx, cy, w, h)
        else:
            ex, ey, ew, eh = self.ema
            self.ema = (alpha * cx + (1 - alpha) * ex,
                        alpha * cy + (1 - alpha) * ey,
                        alpha * w  + (1 - alpha) * ew,
                        alpha * h  + (1 - alpha) * eh)
        return self.ema

    def clamp_ar(self, box: tuple, ar_min: float = 0.9, ar_max: float = 1.3):
        """Clamp the plate box to the allowed aspect-ratio range, keeping the
        centre fixed and expanding only along the deficient dimension.
        Moto plates are near-square; car plates are wide."""
        x1, y1, x2, y2 = box[:4]
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return box
        ar = bw / bh
        if ar_min <= ar <= ar_max:
            return box
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if ar < ar_min:   # too tall → widen
            bw = bh * ar_min
        else:             # too wide → heighten
            bh = bw / ar_max
        # Round outward so the resulting box always satisfies the AR range
        # (int() would shrink both sides and can exceed the bound).
        return (math.floor(cx - bw * 0.5), math.floor(cy - bh * 0.5),
                math.ceil(cx + bw * 0.5), math.ceil(cy + bh * 0.5))

    def set_plate_anchor(self, plate_box: tuple):
        """Store the plate position relative to the current vehicle box."""
        vx1, vy1, vx2, vy2 = self.box
        vw = max(1.0, vx2 - vx1)
        vh = max(1.0, vy2 - vy1)
        px1, py1, px2, py2 = plate_box[:4]
        self.plate_anchor = (
            (px1 + px2) * 0.5 / vw - vx1 / vw,   # normalized centre dx
            (py1 + py2) * 0.5 / vh - vy1 / vh,   # normalized centre dy
            (px2 - px1) / vw,                    # normalized width
            (py2 - py1) / vh,                    # normalized height
        )

    def predict_anchored(self, max_expand: int = 20):
        """
        Plate position predicted from the *current* vehicle box + last known
        normalized offset. Returns None when no anchor has been recorded yet.
        The predicted box is clamped to stay inside the vehicle box expanded
        by 10% — it can never fly off onto trees/roadside objects.
        """
        if self.plate_anchor is None:
            return None
        vx1, vy1, vx2, vy2 = self.box
        vw = max(1.0, vx2 - vx1)
        vh = max(1.0, vy2 - vy1)
        ndx, ndy, nw, nh = self.plate_anchor
        p_w = nw * vw
        p_h = nh * vh
        cx  = vx1 + ndx * vw
        cy  = vy1 + ndy * vh
        expand = min(self.plate.miss_count * 2, max_expand)
        x1 = int(cx - p_w * 0.5 - expand)
        y1 = int(cy - p_h * 0.5 - expand)
        x2 = int(cx + p_w * 0.5 + expand)
        y2 = int(cy + p_h * 0.5 + expand)
        # Clamp inside the vehicle box expanded by 10%
        pad_x = (vx2 - vx1) * 0.10
        pad_y = (vy2 - vy1) * 0.10
        return (max(int(vx1 - pad_x), x1), max(int(vy1 - pad_y), y1),
                min(int(vx2 + pad_x), x2), min(int(vy2 + pad_y), y2))

    def mark_missed(self):
        self.miss_count += 1

    def fallback_rect(self):
        """
        Privacy fallback: bottom strip of the vehicle box. Returns the
        rectangle covering the last `fallback_frac` of the vehicle height,
        expanded by `fallback_pad_frac`. Used when the plate is persistently
        missed (or never found) on a stable vehicle — better to over-blur
        the likely plate zone than to leave a plate exposed.
        """
        x1, y1, x2, y2 = self.box
        h  = y2 - y1
        w  = x2 - x1
        fy1 = y2 - int(h * self.fallback_frac)
        pad_x = int(w * self.fallback_pad_frac)
        pad_y = int((y2 - fy1) * self.fallback_pad_frac)
        return (max(0, x1 - pad_x), max(0, fy1 - pad_y),
                x2 + pad_x, y2 + pad_y)

    def anchor_rect(self, frac: float = 0.45, y_frac: float = 0.70,
                    pad_frac: float = 0.15, base_box: tuple = None,
                    min_side: float = 0.0):
        """
        Anchor-only plate zone for motorcycles: a fixed near-square region
        centred on the rear (centre-bottom) of the vehicle box, sized as a
        fraction of the box height. No plate detection needed — the zone is
        where the plate must be, so coverage is guaranteed at any distance,
        with zero flicker and zero drift (it follows the stable vehicle box).

        base_box: optional box to derive the zone from instead of self.box
        (used to follow velocity-predicted positions while the moto is missed).
        min_side: minimum zone side in pixels — fragmented close-up boxes would
        otherwise shrink the zone below the plate size.
        """
        if base_box is None:
            base_box = self.box
        x1, y1, x2, y2 = base_box
        vw = max(1.0, x2 - x1)
        vh = max(1.0, y2 - y1)
        side  = max(vh * frac, min_side)
        side  = min(side, vw * 1.2)          # don't exceed the vehicle width
        cx = (x1 + x2) * 0.5
        cy = y1 + vh * y_frac
        pad = side * pad_frac
        return (max(0, int(cx - side * 0.5 - pad)),
                max(0, int(cy - side * 0.5 - pad)),
                int(cx + side * 0.5 + pad),
                int(cy + side * 0.5 + pad))

class _VehicleDetections:
    """
    Minimal wrapper that presents vehicle bounding boxes in the format
    BYTETracker.update() expects (supports boolean-mask and integer indexing).
    """
    def __init__(self, xyxy, confs, clss):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confs, dtype=np.float32).reshape(-1)
        self.cls  = np.asarray(clss,  dtype=np.float32).reshape(-1)
        if len(self.xyxy):
            x1, y1, x2, y2 = self.xyxy[:,0], self.xyxy[:,1], self.xyxy[:,2], self.xyxy[:,3]
            self.xywh = np.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], axis=1)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _VehicleDetections(self.xyxy[idx], self.conf[idx], self.cls[idx])

class SceneTracker:
    """
    Frame-level coordinator using BYTETracker for vehicle association.

    Replaces the original greedy-IoU matcher with ByteTrack, which provides:
      - Kalman-filter position prediction  → survives fast camera / subject motion
      - Two-stage matching                 → recovers vehicles that briefly disappear
      - Stable track IDs across gaps       → plate history survives detection drops

    PlateHistory velocity-based gap-fill works on top: when the vehicle detector
    drops a track temporarily, the plate's last known position + velocity is
    extrapolated for up to max_gap_frames frames.
    """

    def __init__(self, max_gap_frames: int = 8, history_frames: int = 15,
                 min_vehicle_conf: float = 0.60, vehicle_iou_thresh: float = 0.30,
                 standalone_min_ar: float = 1.2, standalone_max_ar: float = 6.0,
                 predict_expand_max: int = 20, predict_max_disp: float = 40.0,
                 fallback_enabled: bool = True, fallback_frac: float = 0.40,
                 fallback_pad_frac: float = 0.25, fallback_min_frames: int = 3,
                 ema_alpha: float = 0.6,
                 moto_ar_min: float = 0.9, moto_ar_max: float = 1.3,
                 moto_anchor: bool = True, moto_anchor_frac: float = 0.45,
                 moto_anchor_y: float = 0.70, moto_anchor_pad: float = 0.15,
                 moto_ghost_frames: int = 6,
                 moto_close_frac: float = 0.40, moto_close_conf: float = 0.20,
                 moto_close_zone_w: float = 1.6, moto_edge_px: int = 4,
                 moto_near_frac: float = 0.15, moto_zone_min_side: float = 40.0,
                 moto_anchor_y_max: float = 0.72,
                 moto_plate_conf: float = 0.30,
                 moto_plate_promote_frames: int = 2,
                 moto_plate_hold_frames: int = 15,
                 moto_plate_pad: int = 10,
                 moto_plate_max_drift: float = 14.0,
                 moto_min_blur_box_h_frac: float = 0.10,
                 moto_max_zone_box_frac: float = 0.35,
                 moto_weak_fender_frac: float = 0.92,
                 emit_max_disp: float = 80.0,
                 zone_filter: str = "ema",
                 kf_process_pos: float = 3.0, kf_process_size: float = 1.5,
                 kf_meas_pos: float = 3.0, kf_meas_size: float = 3.0,
                 kf_gate_max: float = 6.0, kf_max_rejects: int = 2,
                 kf_sigma_pad_k: float = 0.5, kf_sigma_pad_max: float = 8.0,
                 kf_pad_decay: float = 1.5, kf_vel_decay: float = 0.85,
                 kf_anchor_meas_scale: float = 5.0):
        from types import SimpleNamespace
        from ultralytics.trackers import BYTETracker

        self.max_gap_frames      = max_gap_frames
        self.history_frames      = history_frames
        self.standalone_min_ar   = standalone_min_ar
        self.standalone_max_ar   = standalone_max_ar
        self.predict_expand_max  = predict_expand_max
        self.predict_max_disp    = predict_max_disp
        self.fallback_enabled    = fallback_enabled
        self.fallback_frac       = fallback_frac      # bottom strip height (fraction of vehicle box height)
        self.fallback_pad_frac   = fallback_pad_frac  # expansion around the strip
        self.fallback_min_frames = fallback_min_frames
        self.ema_alpha           = ema_alpha          # EMA smoothing of plate box
        self.moto_ar_min         = moto_ar_min        # moto plate AR clamp range
        self.moto_ar_max         = moto_ar_max
        self.moto_anchor         = moto_anchor        # anchor-only mode: skip plate
                                                      # detection on motos, blur a
                                                      # fixed rear zone of the box
        self.moto_anchor_frac    = moto_anchor_frac   # zone side as fraction of box height
        self.moto_anchor_y       = moto_anchor_y      # vertical centre (0..1 of box height)
        self.moto_anchor_pad     = moto_anchor_pad    # extra padding around the zone
        # Anti-flicker persistence for anchor-only moto mode: when the vehicle
        # detector drops the motorcycle and its track is finally deleted, the
        # last anchor zone keeps being emitted for this many extra frames,
        # extrapolated with the box velocity, so the blur never blinks out.
        self.moto_ghost_frames   = moto_ghost_frames
        self._ghost_zones: dict  = {}    # tid → (zone_ema, box_vel, frame_idx, close)
        self.frame_idx           = 0
        # Close-range motorcycle handling (moto fills the frame / cut at the
        # edges): a box taller than moto_close_frac of the frame height is
        # treated as "close" — trusted at moto_close_conf, its anchor zone is
        # widened (lean offset), extended to the frame edge it touches, and the
        # velocity extrapolation is frozen (pixel velocity is unreliable at
        # close range and would drag the blur off the plate).
        self.min_vehicle_conf    = min_vehicle_conf
        self.moto_close_frac     = moto_close_frac
        self.moto_close_conf     = moto_close_conf
        self.moto_close_zone_w   = moto_close_zone_w
        self.moto_edge_px        = moto_edge_px
        # Close/medium motorcycles: a box taller than moto_near_frac of the
        # frame is "near" — the YOLO box of a near moto includes the rider's
        # head, pushing the plate below the box centre, and fragmented boxes
        # shrink the anchor zone below the plate size. So for near motos the
        # zone is shifted down (up to moto_anchor_y_max) and never smaller
        # than moto_zone_min_side px.
        self.moto_near_frac      = moto_near_frac
        self.moto_zone_min_side  = moto_zone_min_side
        self.moto_anchor_y_max   = moto_anchor_y_max
        # Plate-confirmed zone upgrade: minimum plate confidence inside a moto
        # box to start promoting the anchor zone onto the detected plate, and
        # how many consecutive frames it must persist before the promotion is
        # applied (anti-flicker: a single-frame plate detection must never
        # yank the blur around).
        self.moto_plate_conf = moto_plate_conf
        self.moto_plate_promote_frames = max(1, moto_plate_promote_frames)
        self.moto_plate_hold_frames = max(1, moto_plate_hold_frames)
        # Skip blur when the motorcycle box is too small to carry a readable
        # plate (distant bikes). Fraction of frame height.
        self.moto_min_blur_box_h_frac = max(0.0, moto_min_blur_box_h_frac)
        # Cap emitted zone area vs vehicle box (stops 19s-style full-bike ovals
        # when a trusted plate size is available).
        self.moto_max_zone_box_frac = max(0.05, moto_max_zone_box_frac)
        # Fender floor only on weak/miss paths (28s coverage without trusted
        # oval inflation).
        self.moto_weak_fender_frac = min(0.98, max(0.5, moto_weak_fender_frac))
        # Clamp on how far the promoted zone centre may move per frame when the
        # plate is missing (dt > 0). Low-conf detections (0.15–0.3) jitter by
        # tens of px between frames; extrapolating that velocity 1:1 yanks the
        # blur off the plate (measured: 35px jump in 1 frame on the 2:14 moto).
        self.moto_plate_max_drift = max(2.0, moto_plate_max_drift)
        # Padding around the promoted/EMA plate zone (px). Uses the global
        # blur padding so the whole plate + anti-aliased edges are covered
        # even when the EMA lags the true plate by a few pixels.
        self.moto_plate_pad = max(4, moto_plate_pad)
        # Hard cap on how far the emitted blur zone may travel per frame from
        # the last zone emitted for the same track (privacy guard). A plate
        # confirmed with high confidence at (x, y) cannot legitimately jump to
        # (x+500, y+500) a few frames later; runaway vehicle boxes, velocity
        # extrapolation and LK latch-ons would otherwise drag the blur onto
        # trees/roadside objects. The allowed distance is emit_max_disp
        # px/frame of elapsed time, capped so stale anchors can still catch up
        # with a real re-detection after a long gap.
        self.emit_max_disp = max(1.0, float(emit_max_disp))
        # Temporal filter for the plate zone. "ema" is the original cascade of
        # two exponential averages plus velocity extrapolation on gap frames;
        # "kalman" runs one constant-velocity filter instead (no lag on moving
        # plates, no lag/overshoot alternation between detected and missed
        # frames, uncertainty-driven padding). Default "ema" keeps existing
        # runs bit-identical.
        self.zone_filter      = str(zone_filter or "ema").lower()
        self.kf_process_pos   = float(kf_process_pos)
        self.kf_process_size  = float(kf_process_size)
        self.kf_meas_pos      = float(kf_meas_pos)
        self.kf_meas_size     = float(kf_meas_size)
        # Mahalanobis distance beyond which a detection is treated as belonging
        # to something else and left out of the filter. Replaces the fixed
        # px/frame caps: the threshold adapts to the current uncertainty.
        self.kf_gate_max      = float(kf_gate_max)
        # Consecutive gated-out detections after which the filter is re-seeded:
        # a plate that keeps showing up somewhere else really has moved there.
        self.kf_max_rejects   = max(1, int(kf_max_rejects))
        # Extra padding while the plate is missing, proportional to the filter's
        # positional uncertainty (px of sigma → px of padding), capped.
        self.kf_sigma_pad_k   = float(kf_sigma_pad_k)
        self.kf_sigma_pad_max = float(kf_sigma_pad_max)
        # How fast the uncertainty padding may retreat (px/frame): the zone
        # widens at once when the plate is lost, then narrows gradually.
        self.kf_pad_decay     = max(0.0, float(kf_pad_decay))
        # Per-frame velocity decay once the plate has been unobserved for a few
        # frames, so a long coast slows to a stop near the vehicle instead of
        # extrapolating a stale velocity across the frame.
        self.kf_vel_decay     = float(kf_vel_decay)
        # Geometric-anchor updates are ~5× noisier than a real plate detection
        # (measured positional spread 12–17 px vs meas_pos=3). Multiplies the
        # filter's measurement std so the anchor nudges the estimate instead of
        # yanking it away from a locked-on plate.
        self.kf_anchor_meas_scale = max(1.0, float(kf_anchor_meas_scale))
        self.debug_zones = bool(os.environ.get("AUTOCLIP_ZONE_DEBUG"))

        # BYTETracker hyperparameters tuned for dashcam / action-cam footage:
        #   track_high_thresh — stage-1: detections above this are matched first
        #   track_low_thresh  — stage-2: weaker detections used to recover lost tracks
        #   new_track_thresh  — minimum conf to start a brand-new track
        #   match_thresh      — max IoU *distance* (= 1 − IoU) for a valid match
        #   track_buffer      — frames BYTETracker holds a lost track before discarding
        bt_args = SimpleNamespace(
            track_high_thresh = min_vehicle_conf,
            track_low_thresh  = max(0.05, min_vehicle_conf * 0.25),
            # Close-range boxes are promoted to min_vehicle_conf in update()
            # (pre-gated to huge boxes only), so a close moto starts its track
            # instantly instead of waiting for a full-confidence detection.
            new_track_thresh  = min_vehicle_conf,
            match_thresh      = 1.0 - vehicle_iou_thresh,    # IoU 0.3 → distance 0.7
            track_buffer      = max(max_gap_frames * 2, 30), # keep lost tracks long enough
            fuse_score        = True,
        )
        self._byte       = BYTETracker(bt_args)
        self._track_dict: dict = {}   # track_id (int) → VehicleTrack

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def tracks(self) -> list:
        """List of currently active VehicleTrack objects (for debug overlay)."""
        return list(self._track_dict.values())

    def _clamp_emit(self, track, zone: tuple) -> tuple:
        """
        Limit how far an emitted blur zone may travel from the last zone
        emitted for the same track. The zone centre is clamped so the blur
        can never move more than emit_max_disp px/frame: a plate confirmed at
        (x, y) cannot legitimately appear at (x+500, y+500) a few frames
        later — the blur glides toward the new position at the allowed speed
        instead of teleporting onto trees/background. The clamped centre
        becomes the new reference, so repeated over-eager predictions stay
        anchored near the last safe position.

        Zones overlapping the last emitted zone (e.g. an anchor zone being
        refined onto the detected plate) pass through untouched — the blur
        never leaves the plate region, so there is nothing to clamp.

        Zone dimensions are preserved; only the centre is moved.
        """
        if len(zone) < 4:
            return zone
        x1, y1, x2, y2 = zone[:4]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        if track.last_emit is None:
            track.last_emit = (self.frame_idx, cx, cy)
            return zone
        lf, lx, ly = track.last_emit
        if _overlaps((x1, y1, x2, y2),
                     (int(lx - (x2 - x1) * 0.5), int(ly - (y2 - y1) * 0.5),
                      int(lx + (x2 - x1) * 0.5), int(ly + (y2 - y1) * 0.5))):
            track.last_emit = (self.frame_idx, cx, cy)
            return zone
        dt = min(max(1, self.frame_idx - lf), self.max_gap_frames)
        allowed = self.emit_max_disp * dt
        dx = cx - lx
        dy = cy - ly
        m = math.hypot(dx, dy)
        if m <= allowed:
            track.last_emit = (self.frame_idx, cx, cy)
            return zone
        f = allowed / m
        ncx = lx + dx * f
        ncy = ly + dy * f
        w = x2 - x1
        h = y2 - y1
        track.last_emit = (self.frame_idx, ncx, ncy)
        return (int(ncx - w * 0.5), int(ncy - h * 0.5),
                int(ncx + w * 0.5), int(ncy + h * 0.5))

    def _kf_advance(self, track) -> None:
        """Predict the track's filter forward to the current frame, once."""
        if track.kf is None:
            return
        dt = self.frame_idx - track.kf_frame
        if dt > 0:
            track.kf.predict(dt)
            track.kf_frame = self.frame_idx

    def _kf_new(self, track, box, conf: float) -> None:
        track.kf = BoxFilter(
            box, conf,
            process_pos=self.kf_process_pos, process_size=self.kf_process_size,
            meas_pos=self.kf_meas_pos, meas_size=self.kf_meas_size,
            vel_decay=self.kf_vel_decay,
        )
        track.kf_frame   = self.frame_idx
        track.kf_rejects = 0

    def _kf_observe(self, track, box, conf: float,
                    noise_scale: float = 1.0, gated: bool = True) -> None:
        """Feed a plate box to the track's filter, creating it if new.

        A detection too far from the prediction (gate) is ignored rather than
        applied, so the estimate keeps coasting instead of being yanked onto a
        different object. Two rejections in a row mean the plate really did move
        somewhere else (a close vehicle accelerating past, or the track latching
        onto a different plate), so the filter is re-seeded from the detection
        rather than coasting on a stale position for the whole hold window.

        *noise_scale* > 1 marks a weak (non-detector) measurement — typically the
        geometric motorcycle anchor. *gated*=False skips the Mahalanobis gate:
        an anchor derived from the vehicle box is our own guess, not a possibly-
        wrong detector association, so rejecting it would leave the filter without
        any update on frames that have no plate detection.
        """
        if track.kf is None:
            self._kf_new(track, box, conf)
            return
        self._kf_advance(track)
        if (not gated) or track.kf.gate_distance(box) <= self.kf_gate_max:
            track.kf.update(box, conf, noise_scale=noise_scale)
            track.kf_rejects = 0
        else:
            track.kf_rejects += 1
            if track.kf_rejects >= self.kf_max_rejects:
                self._kf_new(track, box, conf)

    def _moto_too_small(self, box, fh: int) -> bool:
        """True if the vehicle box is too small for a readable plate → no blur."""
        if self.moto_min_blur_box_h_frac <= 0 or fh <= 0:
            return False
        return (box[3] - box[1]) < self.moto_min_blur_box_h_frac * fh

    def _moto_trusted(self, track) -> bool:
        """Plate pose is promoted and still fresh → emit a tight zone."""
        if track.plate_rel is None:
            return False
        if track.moto_plate_streak < self.moto_plate_promote_frames:
            return False
        # Taillight-height poses must not lock a tight oval (28s regression).
        _rx, ry, _rw, _rh = track.plate_rel
        if ry < 0.40:
            return False
        return track.moto_plate_miss <= max(2, self.moto_plate_promote_frames)

    def _plate_score(self, plate, box) -> float:
        """Rank in-box plate candidates: prefer mid-rear over taillight/wheel."""
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        cx = (plate[0] + plate[2]) * 0.5
        cy = (plate[1] + plate[3]) * 0.5
        rx = (cx - box[0]) / bw
        ry = (cy - box[1]) / bh
        conf = float(plate[4]) if len(plate) > 4 else 0.0
        # Peak preference around mid-horizontal, mid-lower rear.
        score = conf
        score -= abs(rx - 0.50) * 0.6
        score -= abs(ry - 0.55) * 0.8
        if ry < 0.35:
            score -= 0.4
        if rx < 0.25 or rx > 0.75:
            score -= 0.5
        return score

    def _cap_moto_zone(self, zone, box) -> tuple:
        """Shrink *zone* toward its centre if it covers too much of *box*."""
        max_frac = self.moto_max_zone_box_frac
        if max_frac <= 0:
            return zone
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        zw = max(1.0, zone[2] - zone[0])
        zh = max(1.0, zone[3] - zone[1])
        if (zw * zh) / (bw * bh) <= max_frac:
            return zone
        scale = ((max_frac * bw * bh) / (zw * zh)) ** 0.5
        cx = (zone[0] + zone[2]) * 0.5
        cy = (zone[1] + zone[3]) * 0.5
        nw, nh = zw * scale, zh * scale
        return (int(cx - nw * 0.5), int(cy - nh * 0.5),
                int(cx + nw * 0.5), int(cy + nh * 0.5))

    def _moto_anchor_box(self, track, base_box=None, fw: int = 0, fh: int = 0) -> tuple:
        """Geometric plate guess for a motorcycle, sized to the measured plate.

        Trusted path (promoted plate_rel): tight box around the learned pose —
        no fender floor (fixes 19s full-bike ovals).
        Weak / no-plate path: larger safety zone + fender floor (covers 28s).
        """
        if base_box is None:
            base_box = track.box
        x1, y1, x2, y2 = base_box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        trusted = self._moto_trusted(track)
        if track.plate_rel is not None and trusted:
            rx, ry, rw, rh = track.plate_rel
            side_w = max(bw * rw * 1.20, bw * 0.14, self.moto_zone_min_side * 0.4)
            side_h = max(bh * rh * 1.20, bh * 0.14, self.moto_zone_min_side * 0.4)
            cx = x1 + bw * rx
            cy = y1 + bh * ry
            zone = (int(cx - side_w * 0.5), int(cy - side_h * 0.5),
                    int(cx + side_w * 0.5), int(cy + side_h * 0.5))
            zone = self._cap_moto_zone(zone, base_box)
        elif track.plate_rel is not None:
            rx, ry, rw, rh = track.plate_rel
            ry = min(0.72, ry + 0.05)
            side_w = max(bw * rw * 1.30, bw * 0.20, self.moto_zone_min_side * 0.5)
            side_h = max(bh * rh * 1.30, bh * 0.26, self.moto_zone_min_side * 0.5)
            cx = x1 + bw * rx
            cy = y1 + bh * ry
            z2 = max(int(cy + side_h * 0.5), int(y1 + bh * self.moto_weak_fender_frac))
            z1 = min(int(cy - side_h * 0.5), z2 - int(side_h))
            zone = (int(cx - side_w * 0.5), z1, int(cx + side_w * 0.5), z2)
        else:
            near = fh > 0 and bh > self.moto_near_frac * fh
            y_frac = self.moto_anchor_y
            if near:
                y_frac = min(self.moto_anchor_y_max,
                             self.moto_anchor_y + 0.20 * bh / max(fh, 1))
            min_side = max(bh * 0.28,
                           self.moto_zone_min_side if bh < self.moto_zone_min_side * 2
                           else 0.0)
            zone = track.anchor_rect(self.moto_anchor_frac, y_frac,
                                     self.moto_anchor_pad,
                                     base_box=base_box, min_side=min_side)
            zx1, zy1, zx2, zy2 = zone
            zy2 = max(zy2, int(y1 + bh * self.moto_weak_fender_frac))
            zone = (zx1, zy1, zx2, zy2)
            if near:
                cx = (zone[0] + zone[2]) * 0.5
                nw = (zone[2] - zone[0]) * self.moto_close_zone_w
                zone = (int(cx - nw * 0.5), zone[1], int(cx + nw * 0.5), zone[3])
        zx1, zy1, zx2, zy2 = zone
        if fw and fh:
            zx1 = max(0, zx1); zy1 = max(0, zy1)
            zx2 = min(zx2, fw - 1); zy2 = min(zy2, fh - 1)
            if base_box[0] <= self.moto_edge_px:
                zx1 = 0
            if base_box[2] >= fw - self.moto_edge_px:
                zx2 = max(zx2, fw - 1)
            if base_box[1] <= self.moto_edge_px:
                zy1 = 0
            if base_box[3] >= fh - self.moto_edge_px:
                zy2 = max(zy2, fh - 1)
        return (zx1, zy1, zx2, zy2)

    def _plate_geometry_ok(self, plate, box) -> bool:
        """True if *plate* sits where a motorcycle plate can physically be.

        Rejects false positives that yank the filter onto a wheel, exhaust tip,
        rider backpack or road patch: measured plates on the ref clip live at
        roughly (0.3–0.7, 0.30–0.55) of the vehicle box with size ~0.15–0.40 of
        box width. Anything far outside that envelope is noise, not a plate —
        the safety net still blurs it as a patch, but it must not drive the
        smoothed zone.
        """
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        cx = (plate[0] + plate[2]) * 0.5
        cy = (plate[1] + plate[3]) * 0.5
        rx = (cx - box[0]) / bw
        ry = (cy - box[1]) / bh
        rw = (plate[2] - plate[0]) / bw
        rh = (plate[3] - plate[1]) / bh
        if not (0.22 <= rx <= 0.78):
            return False
        # Italian moto plates can sit very low on the rear fender (ry≈0.70 on
        # tall adventure bikes); the old 0.62 cap rejected the real plate and
        # left the filter locked on the brighter taillight above it.
        # Floor at 0.32 rejects the brighter taillight band that stole the
        # filter at 28s while the real plate sat lower on the fender.
        if not (0.38 <= ry <= 0.78):
            return False
        if rw < 0.08 or rh < 0.05:
            return False
        if rw > 0.60 or rh > 0.55:
            return False
        return True

    def _learn_plate_rel(self, track, plate) -> None:
        """EMA-update the track's relative plate pose from a trusted detection."""
        box = track.box
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        rx = ((plate[0] + plate[2]) * 0.5 - box[0]) / bw
        ry = ((plate[1] + plate[3]) * 0.5 - box[1]) / bh
        rw = (plate[2] - plate[0]) / bw
        rh = (plate[3] - plate[1]) / bh
        if track.plate_rel is None:
            track.plate_rel = (rx, ry, rw, rh)
            return
        a = 0.35
        prx, pry, prw, prh = track.plate_rel
        track.plate_rel = (a * rx + (1 - a) * prx,
                           a * ry + (1 - a) * pry,
                           a * rw + (1 - a) * prw,
                           a * rh + (1 - a) * prh)

    def _moto_min_side(self, track) -> float:
        """Zone side the geometric EMA-path anchor would use for this box.

        Kept for the EMA fallback path and for ghost-zone seeding. The Kalman
        path no longer applies this as an emit floor: size comes from the filter
        estimate plus uncertainty padding.
        """
        x1, y1, x2, y2 = track.box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        return min(max(self.moto_zone_min_side, bh * 0.32), bw * 1.2)

    def _kf_zone(self, track, pad: int, min_side: float = 0.0) -> tuple:
        """Zone from the filter: estimate plus padding grown by uncertainty.

        The uncertainty term is applied with hysteresis — it grows immediately
        (coverage first) but retreats by at most kf_pad_decay px per frame.
        Without it the zone would breathe visibly, widening on every gap frame
        and snapping back on every detection.

        *min_side*, when > 0, floors the emitted side length. The Kalman moto
        path no longer passes it: size comes from the filter estimate, and
        uncovered detections are patched by the safety net instead.
        """
        extra = min(self.kf_sigma_pad_max,
                    self.kf_sigma_pad_k * track.kf.position_sigma())
        extra = max(extra, track.kf_pad - self.kf_pad_decay)
        track.kf_pad = extra
        zone = track.kf.box(pad=pad + extra)
        if min_side > 0:
            cx, cy = track.kf.centre
            half_w = max((zone[2] - zone[0]) * 0.5, min_side * 0.5)
            half_h = max((zone[3] - zone[1]) * 0.5, min_side * 0.5)
            zone = (int(cx - half_w), int(cy - half_h),
                    int(cx + half_w), int(cy + half_h))
        return zone

    def _kf_emit(self, effective, track, track_id, pad: int, conf: float,
                 source: str, measured=None, clamp_box=None,
                 min_side: float = 0.0) -> None:
        """Emit the filtered zone, plus any detection it does not cover.

        Privacy safety net: whenever the filter and the detector disagree — a
        gated-out detection, an estimate still catching up with a fast plate, or
        a second plate inside the same vehicle box (two motorcycles riding side
        by side produce overlapping boxes) — each uncovered box is emitted as its
        own zone. A smoothed zone may lag; a plate the detector could read is
        never left exposed. The tracked zone keeps its own smooth geometry, so
        this costs no size jump.

        *measured* is a single box or a list of boxes.
        """
        zone = self._kf_zone(track, pad, min_side)
        if clamp_box is not None:
            zone = _clamp_to_box(zone, clamp_box)
            trusted = track.cls == 3 and self._moto_trusted(track)
            # Cap the smoothed estimate first. Fender stretch must come AFTER
            # the cap — otherwise the 19s size limit undoes the 28s coverage
            # and leaves the real plate exposed under a tight taillight/tire oval.
            if track.cls == 3 and trusted:
                zone = self._cap_moto_zone(zone, clamp_box)
            if track.cls == 3 and not trusted:
                bx1, by1, bx2, by2 = clamp_box
                fender = int(by1 + (by2 - by1) * self.moto_weak_fender_frac)
                if zone[3] < fender:
                    zone = (zone[0], zone[1], zone[2], fender)
                # Keep horizontal span from covering the whole bike, but never
                # shrink the vertical fender coverage we just added.
                zw = zone[2] - zone[0]
                max_w = max(1.0, (bx2 - bx1) * 0.55)
                if zw > max_w:
                    cx = (zone[0] + zone[2]) * 0.5
                    zone = (int(cx - max_w * 0.5), zone[1],
                            int(cx + max_w * 0.5), zone[3])
                # Wheel FPs pull the filter to the side — re-centre horizontally
                # on the vehicle so the fender band covers the real plate.
                bcx = (bx1 + bx2) * 0.5
                zcx = (zone[0] + zone[2]) * 0.5
                if abs(zcx - bcx) > (bx2 - bx1) * 0.12:
                    shift = bcx - zcx
                    zone = (int(zone[0] + shift), zone[1],
                            int(zone[2] + shift), zone[3])
                    zone = _clamp_to_box(zone, clamp_box)
                    if zone[3] < fender:
                        zone = (zone[0], zone[1], zone[2], fender)
        # Bound how fast the emitted zone may travel. This only bites when the
        # new zone does not overlap the previous one — i.e. on a genuine
        # discontinuity, such as the hold window expiring and the zone reverting
        # to the geometric anchor — so normal tracking passes through unfiltered
        # and only teleports are turned into a short glide.
        zone = self._clamp_emit(track, zone)
        track.plate_zone_prev = track.kf.centre
        effective.append((*zone, conf, source, track_id))
        if measured is None:
            return
        boxes = measured if isinstance(measured, (list, tuple)) \
            and measured and isinstance(measured[0], (list, tuple)) else [measured]
        for box in boxes:
            if covered_fraction(box[:4], zone) < 0.9:
                # Tagged 'patch' so it is distinguishable from the smoothed zone:
                # it is a guaranteed-coverage rectangle, not part of the track's
                # trajectory, and must not be quad-refined or measured as jitter.
                effective.append((*box[:4], conf, "patch", track_id))

    def update(self, all_vehicles: list, plate_rects: list,
               frame: np.ndarray = None, frame_size: tuple = None) -> list:
        """
        all_vehicles : [(cls, x1, y1, x2, y2, conf), ...]
        plate_rects  : [(x1, y1, x2, y2, conf), ...]  — current frame detections
        frame        : optional BGR frame; enables LK optical-flow gap-fill when given
        frame_size   : optional (width, height) of the frame in pixels; enables
                       the close-range motorcycle handling (size-adaptive conf
                       acceptance, frozen ghost, edge-extended anchor zone)

        Returns effective plate list: detected plates + gap-filled predicted plates.
        Predicted entries carry conf = -1.0 so the debug overlay can label them.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        fw, fh = frame_size if frame_size is not None else (0, 0)

        def _is_close(box):
            return fh > 0 and (box[3] - box[1]) > self.moto_close_frac * fh

        def _widen_zone(zone, factor):
            x1, y1, x2, y2 = zone
            cx = (x1 + x2) * 0.5
            nw = (x2 - x1) * factor
            return (int(cx - nw * 0.5), y1, int(cx + nw * 0.5), y2)

        def _extend_to_edge(zone, base_box):
            x1, y1, x2, y2 = zone
            # Clamp inside the frame first — scaled zones may overshoot the
            # edges; then extend only up to the frame boundary.
            if fw and fh:
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(x2, fw - 1); y2 = min(y2, fh - 1)
            if base_box[0] <= self.moto_edge_px:
                x1 = 0
            if base_box[2] >= fw - self.moto_edge_px:
                x2 = max(x2, fw - 1)
            if base_box[1] <= self.moto_edge_px:
                y1 = 0
            if base_box[3] >= fh - self.moto_edge_px:
                y2 = max(y2, fh - 1)
            return (x1, y1, x2, y2)

        # ── 1. Feed vehicle detections to BYTETracker ─────────────────────────
        # Size-adaptive confidence gate: a box that fills more than
        # moto_close_frac of the frame height is a near-certain close-up and is
        # trusted down to moto_close_conf — promoted to min_vehicle_conf so
        # ByteTrack starts/keeps its track instantly. Everything else keeps the
        # original stage-2 floor (no regression for weak distant detections).
        if all_vehicles:
            low  = max(0.05, self.min_vehicle_conf * 0.25)
            gated = []
            for (cls, x1, y1, x2, y2, conf) in all_vehicles:
                if conf >= self.moto_close_conf and _is_close((x1, y1, x2, y2)):
                    gated.append((cls, x1, y1, x2, y2, max(conf, self.min_vehicle_conf)))
                elif conf >= low:
                    gated.append((cls, x1, y1, x2, y2, conf))
            all_vehicles = gated
            xyxy  = np.array([[x1, y1, x2, y2] for (_, x1, y1, x2, y2, _) in all_vehicles],
                             dtype=np.float32)
            confs = np.array([c   for (*_, c)          in all_vehicles], dtype=np.float32)
            clss  = np.array([cls for (cls, *_)         in all_vehicles], dtype=np.float32)
            det   = _VehicleDetections(xyxy, confs, clss)
        else:
            det   = _VehicleDetections(np.zeros((0, 4)), [], [])

        # BYTETracker returns active tracks: rows of [x1,y1,x2,y2, id,conf,cls,idx]
        active = self._byte.update(det)

        # ultralytics 8.4.117: STrack.activate() sets is_activated=True only on
        # frame 1 — tracks born on later frames stay "unconfirmed" and are
        # silently dropped by _format_output(). They still carry THIS frame's
        # real detections, so promote them to active rows: a re-born moto
        # track must be visible on its first frame back, or the ghost-seed
        # (anti-flicker zone handover) can never trigger.
        from ultralytics.trackers.byte_tracker import TrackState
        _unconf = [t for t in self._byte.tracked_stracks
                   if not t.is_activated and t.state == TrackState.Tracked
                   and t.frame_id == self._byte.frame_id]
        if _unconf:
            _rows = np.asarray(
                [[float(t.xyxy[0]), float(t.xyxy[1]), float(t.xyxy[2]), float(t.xyxy[3]),
                  float(t.track_id), float(t.score), float(t.cls), float(t.idx)]
                 for t in _unconf],
                dtype=np.float32,
            )
            active = _rows if len(active) == 0 else np.vstack([active, _rows])

        # ── 2. Sync our VehicleTrack dict with BYTETracker's output ───────────
        # Note: row[7] (detection index) is NOT reliable for distinguishing
        # Kalman-predicted tracks — ByteTrack keeps the last matched idx on
        # tracks it extrapolates. We instead check whether the returned box
        # matches a real detection of THIS frame (IoU), so a close-up moto
        # never moves on pure Kalman prediction.
        det_boxes = np.asarray(
            [[x1, y1, x2, y2] for (_, x1, y1, x2, y2, _) in all_vehicles],
            dtype=np.float64,
        ) if all_vehicles else np.zeros((0, 4), dtype=np.float64)

        def _is_real_detection(box: tuple) -> bool:
            if len(det_boxes) == 0:
                return False
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            inter_x1 = np.maximum(x1, det_boxes[:, 0])
            inter_y1 = np.maximum(y1, det_boxes[:, 1])
            inter_x2 = np.minimum(x2, det_boxes[:, 2])
            inter_y2 = np.minimum(y2, det_boxes[:, 3])
            iw = np.maximum(0, inter_x2 - inter_x1)
            ih = np.maximum(0, inter_y2 - inter_y1)
            inter = iw * ih
            dw = np.maximum(bw, det_boxes[:, 2] - det_boxes[:, 0])
            dh = np.maximum(bh, det_boxes[:, 3] - det_boxes[:, 1])
            union = dw * dh
            return bool(np.any(inter / np.maximum(union, 1e-6) > 0.5))

        active_ids = set()
        for row in active:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            tid = int(row[4])
            active_ids.add(tid)

            if tid not in self._track_dict:
                vt    = VehicleTrack((x1, y1, x2, y2), self.frame_idx,
                                     self.history_frames,
                                     cls=int(row[6]) if len(row) > 6 else 0)
                vt.id = tid   # use BYTETracker's stable ID instead of the counter
                # Anti-flicker: a re-born moto track starts with the zone it
                # had when it died (ghost seed) instead of from scratch, so the
                # blur doesn't jump/blink when the detector drops and
                # re-acquires the same motorcycle.
                if (self.moto_anchor and vt.cls == 3 and self._ghost_zones):
                    new_box = (x1, y1, x2, y2)
                    for gid, (ze, _, _, _) in list(self._ghost_zones.items()):
                        gx1 = int(ze[0] - ze[2] * 0.5)
                        gy1 = int(ze[1] - ze[3] * 0.5)
                        gx2 = int(ze[0] + ze[2] * 0.5)
                        gy2 = int(ze[1] + ze[3] * 0.5)
                        if _overlaps(new_box, (gx1, gy1, gx2, gy2)):
                            vt.zone_ema = ze
                            vt.zone_ghost_seeded = True
                            del self._ghost_zones[gid]
                            break
                self._track_dict[tid] = vt
            else:
                # Predicted (Kalman) positions don't move the box: close-up
                # motos must stay glued to the last real detection.
                self._track_dict[tid].update_box(
                    (x1, y1, x2, y2), self.frame_idx,
                    detected=_is_real_detection((x1, y1, x2, y2)),
                )

        # Tracks not returned this frame are missed; drop after gap-fill window
        for tid in list(self._track_dict):
            if tid not in active_ids:
                self._track_dict[tid].mark_missed()
                if self._track_dict[tid].miss_count > self.max_gap_frames:
                    track = self._track_dict.pop(tid)
                    # Persist the last anchor zone after the track dies so the
                    # blur doesn't blink out between re-detections. Only for
                    # anchor-mode motos — other vehicles have plate history.
                    kf_seeded = (self.zone_filter == "kalman"
                                 and track.kf is not None)
                    if (self.moto_anchor and track.cls == 3
                            and (track.zone_ema is not None or kf_seeded)):
                        # Close motos: freeze the ghost in place — velocity
                        # extrapolation at close range drags the blur off the
                        # (nearly static) plate, so no velocity is stored.
                        last_box = track.last_detected_box or track.box
                        close    = _is_close(last_box)
                        if kf_seeded:
                            # Continue from where the filter actually left the
                            # zone. zone_ema is box-derived and goes unused while
                            # the filter drives the zone, so seeding the ghost
                            # from it made the blur jump hundreds of px the
                            # moment the track died.
                            gx1, gy1, gx2, gy2 = self._kf_zone(track,
                                                               self.moto_plate_pad)
                            seed = ((gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5,
                                    gx2 - gx1, gy2 - gy1)
                            vel  = None if close else track.kf.velocity
                        else:
                            seed = track.zone_ema
                            vel  = None if close else track.box_vel
                        self._ghost_zones[tid] = (seed, vel, self.frame_idx, close)

        # ── 3. AR-filter standalone plates ────────────────────────────────────
        # Plates overlapping any current or last-known vehicle zone pass through.
        # Standalone plates (outside all vehicle boxes) must have a realistic AR.
        vehicle_zones = [(v[1], v[2], v[3], v[4]) for v in all_vehicles] + \
                        [t.box for t in self._track_dict.values()]
        filtered = []
        for p in plate_rects:
            if any(_overlaps(p[:4], z) for z in vehicle_zones):
                filtered.append(p)
            else:
                x1, y1, x2, y2 = p[:4]
                h = y2 - y1
                if h > 0 and self.standalone_min_ar <= (x2 - x1) / h <= self.standalone_max_ar:
                    filtered.append(p)
        plate_rects = filtered

        # ── 4. Plate recording + gap-fill (no suppression) ───────────────────────
        #
        # We blur EVERY distinct candidate that survives the confidence and AR
        # filters.  "One plate per vehicle" suppression has been removed because:
        #
        #   • Missing a real plate (privacy failure) is always worse than blurring
        #     an extra region (cosmetic issue).
        #   • A persistent false positive that keeps winning by raw confidence
        #     would otherwise permanently shadow the real plate.
        #
        # merge_overlapping() in detect_plates() still collapses near-identical
        # detections of the *same* plate produced by overlapping SAHI tiles, so
        # we don't blur the same region multiple times.
        #
        # History-aware trajectory recording (for gap-fill only):
        #   Even though we blur all candidates, the PlateHistory still needs a
        #   single position to track for velocity-based gap-fill prediction.
        #   We pick the candidate most consistent with the established trajectory:
        #   - No history yet            → highest confidence
        #   - ≥ GATE_HISTORY confirmed  → prefer overlap with predicted zone;
        #                                 fall back to highest confidence if none
        #                                 overlap (fast vehicle, bad prediction)
        _GATE_HISTORY = 3

        effective = []
        claimed   = set()   # indices of plate_rects already added to effective

        # ── 4a. Ghost anchor zones (anti-flicker persistence) ────────────────
        # Motorcycles whose track died keep a virtual anchor zone for
        # moto_ghost_frames, extrapolated with the recorded box velocity.
        # Skipped when a live track now covers the same area (re-detected).
        live_moto_boxes = [t.box for t in self._track_dict.values() if t.cls == 3]
        for tid in list(self._ghost_zones):
            zone_ema, box_vel, born, close = self._ghost_zones[tid]
            if self.frame_idx - born > self.moto_ghost_frames:
                del self._ghost_zones[tid]
                continue
            ex, ey, ew, eh = zone_ema
            cx, cy = ex, ey
            if box_vel is not None:
                dt = self.frame_idx - born
                vx, vy = box_vel
                if abs(vx) <= self.predict_max_disp and abs(vy) <= self.predict_max_disp:
                    cx = ex + vx * dt
                    cy = ey + vy * dt
            ghost = (int(cx - ew * 0.5), int(cy - eh * 0.5),
                     int(cx + ew * 0.5), int(cy + eh * 0.5))
            if close:
                ghost = _extend_to_edge(ghost, ghost)
            if any(_overlaps(ghost, b) for b in live_moto_boxes):
                del self._ghost_zones[tid]   # live track took over — drop ghost
                continue
            effective.append((*ghost, -2.0, "anchor", tid))

        for track_id, track in self._track_dict.items():
            # Anchor-only mode for motorcycles: skip plate detection entirely
            # and blur a fixed rear zone of the vehicle box. Guaranteed coverage
            # at any distance with zero flicker/drift — the plate zone is where
            # the plate must be, by geometry.
            if self.moto_anchor and track.cls == 3:
                if self._moto_too_small(track.box, fh):
                    # Distant / unreadable: swallow in-box plates, emit nothing.
                    indexed = [(i, p) for i, p in enumerate(plate_rects)
                               if i not in claimed and _overlaps(p[:4], track.box)]
                    for i, _ in indexed:
                        claimed.add(i)
                    continue
                indexed = [(i, p) for i, p in enumerate(plate_rects)
                           if i not in claimed and _overlaps(p[:4], track.box)]
                for i, _ in indexed:
                    claimed.add(i)   # swallow detections so they don't double-blur
                if self.debug_zones:
                    best0 = max(indexed, key=lambda ip: ip[1][4], default=None)
                    plconf = f"{best0[1][4]:.2f}" if best0 else "--"
                    print(f"[TRK] f={self.frame_idx} tid={track_id} "
                          f"box=({track.box[0]:.0f},{track.box[1]:.0f},"
                          f"{track.box[2]:.0f},{track.box[3]:.0f}) "
                          f"streak={track.moto_plate_streak} miss={track.moto_plate_miss} "
                          f"nplates={len(indexed)} plconf={plconf}", flush=True)

                # Plate-confirmed zone upgrade: when the plate model reliably
                # finds the plate inside this moto (conf >= moto_plate_conf for
                # moto_plate_promote_frames consecutive frames), the geometric
                # anchor is replaced by the EMA-smoothed plate detection —
                # precise at distance (23-30s case), zero drift, no anchor
                # estimation error. Once promoted, the zone is HELD and
                # extrapolated with the recorded plate velocity across
                # detection gaps (angled motos only show the plate in bursts),
                # and only falls back to the anchor after
                # moto_plate_hold_frames frames without a detection.
                best_plate = None
                if indexed:
                    geo_candidates = [
                        (i, p) for i, p in indexed
                        if len(p) > 4 and p[4] >= self.moto_plate_conf
                        and self._plate_geometry_ok(p, track.box)
                    ]
                    if geo_candidates:
                        _, best_plate = max(
                            geo_candidates,
                            key=lambda ip: self._plate_score(ip[1], track.box),
                        )
                    else:
                        _, best_plate = max(
                            indexed,
                            key=lambda ip: ip[1][4] if len(ip[1]) > 4 else 1.0,
                        )
                if best_plate is not None and best_plate[4] >= self.moto_plate_conf:
                    geo_ok = self._plate_geometry_ok(best_plate, track.box)
                    if geo_ok:
                        track.moto_plate_streak += 1
                        track.moto_plate_miss = 0
                        track.plate_last_seen = self.frame_idx
                        track.ema_update(best_plate[:4], self.ema_alpha)
                        self._learn_plate_rel(track, best_plate)
                        # Plate-centre velocity from consecutive detections (EMA).
                        cx, cy, _cw, _ch = track.ema
                        if track.plate_prev is not None:
                            vx = cx - track.plate_prev[0]
                            vy = cy - track.plate_prev[1]
                            pvx, pvy = track.plate_vel
                            track.plate_vel = (0.6 * vx + 0.4 * pvx,
                                               0.6 * vy + 0.4 * pvy)
                        track.plate_prev = (cx, cy)
                        if self.zone_filter == "kalman":
                            # Only geometrically-plausible detections drive the
                            # filter. Outliers (wheel, backpack, road) still reach
                            # the safety net below as patches, so privacy holds
                            # without yanking the smoothed zone off the plate.
                            self._kf_observe(track, best_plate[:4],
                                             float(best_plate[4]))
                            self._kf_emit(effective, track, track_id,
                                          self.moto_plate_pad, -2.0, "anchor",
                                          measured=[p for _, p in indexed],
                                          clamp_box=track.box)
                            continue
                        # EMA path (unchanged): emit immediately on detection.
                        if track.ema is not None:
                            pcx, pcy, pw, ph = track.ema
                            pad = self.moto_plate_pad
                            zone = (max(0, int(pcx - pw * 0.5) - pad),
                                    max(0, int(pcy - ph * 0.5) - pad),
                                    int(pcx + pw * 0.5) + pad,
                                    int(pcy + ph * 0.5) + pad)
                            track.zone_ema_update(zone, self.ema_alpha)
                            ex, ey, ew, eh = track.zone_ema
                            zone = (int(ex - ew * 0.5), int(ey - eh * 0.5),
                                    int(ex + ew * 0.5), int(ey + eh * 0.5))
                            track.plate_zone_prev = (ex, ey)
                            zone = self._cap_moto_zone(zone, track.box)
                            effective.append((*self._clamp_emit(track, zone),
                                              -2.0, "anchor", track_id))
                            continue
                    elif self.zone_filter == "kalman" and track.kf is not None:
                        # Implausible detection: do not update the filter, but
                        # still emit the current estimate and patch the outlier
                        # so it is never left exposed.
                        self._kf_advance(track)
                        self._kf_emit(effective, track, track_id,
                                      self.moto_plate_pad, -2.0, "anchor",
                                      measured=[p for _, p in indexed],
                                      clamp_box=track.box)
                        continue
                    else:
                        track.moto_plate_miss += 1
                else:
                    track.moto_plate_miss += 1
                    if track.moto_plate_miss >= self.moto_plate_hold_frames:
                        # Hold expired: give up the promotion, back to anchor.
                        track.moto_plate_streak = 0
                        track.plate_prev = None

                if (self.zone_filter == "kalman" and track.kf is not None
                        and 0 <= self.frame_idx - track.plate_last_seen
                        <= self.moto_plate_hold_frames):
                    # Detection missing but still inside the hold window: coast
                    # on the filter. The padding grows with the filter's
                    # uncertainty, so a longer gap widens the zone instead of
                    # letting the plate slip out of it. No geometric pull here —
                    # a locked-on plate estimate must not be dragged back to the
                    # box centre while the plate is merely between detections.
                    self._kf_advance(track)
                    self._kf_emit(effective, track, track_id,
                                  self.moto_plate_pad, -2.0, "anchor",
                                  clamp_box=track.box,
                                  measured=[p for _, p in indexed] or None)
                    continue

                if self.zone_filter == "kalman":
                    # No plate (or hold expired): feed a weak measurement.
                    # Prefer the track's learned relative pose (where this bike's
                    # plate actually sits) over the generic geometric guess —
                    # that is what stops the blur falling onto the tire at 24s
                    # or drifting onto the rider's back at 28s.
                    if track.frames_seen < self.fallback_min_frames:
                        # Still claim→patch so a newborn track that swallowed a
                        # detection never leaves that plate exposed.
                        for _, p in indexed:
                            effective.append((*p[:4],
                                              p[4] if len(p) > 4 else -2.0,
                                              "patch", track_id))
                        continue
                    close = _is_close(track.box)
                    base_box = track.box
                    if track.miss_count > 0 and not close:
                        pred = track.predict_box(self.frame_idx,
                                                 self.predict_max_disp)
                        if pred is not None:
                            base_box = pred
                    elif close and track.last_detected_frame < self.frame_idx:
                        base_box = track.last_detected_box
                    if track.zone_ghost_seeded and track.zone_ema is not None:
                        ex, ey, ew, eh = track.zone_ema
                        zone = (int(ex - ew * 0.5), int(ey - eh * 0.5),
                                int(ex + ew * 0.5), int(ey + eh * 0.5))
                        zone = _extend_to_edge(zone, base_box)
                        track.zone_ghost_seeded = False
                        effective.append((*self._clamp_emit(track, zone),
                                          -2.0, "anchor", track_id))
                        continue
                    anchor = self._moto_anchor_box(track, base_box, fw, fh)
                    # Learned relative pose is ~2× noisier than a real detection;
                    # the generic geometric guess stays at kf_anchor_meas_scale.
                    scale = (max(2.0, self.kf_anchor_meas_scale * 0.4)
                             if track.plate_rel is not None
                             else self.kf_anchor_meas_scale)
                    self._kf_observe(track, anchor, conf=1.0,
                                     noise_scale=scale, gated=False)
                    self._kf_emit(effective, track, track_id,
                                  self.moto_plate_pad, -2.0, "anchor",
                                  measured=[p for _, p in indexed] or None,
                                  clamp_box=track.box)
                    continue

                if track.frames_seen >= self.fallback_min_frames:
                    dt = self.frame_idx - track.plate_last_seen
                    # Hold of ANY detection: even a single in-box plate hit
                    # (streak=1) keeps the EMA zone extrapolated across the gap
                    # — private-by-default. Requiring promote_frames here made
                    # the very next detection-less frame fall back to the
                    # geometric anchor, which misses the plate at distance.
                    if (track.ema is not None
                            and 0 <= dt <= self.moto_plate_hold_frames):
                        # Promotion active, detection missing this frame:
                        # zone = EMA-smoothed plate box plus a small pad,
                        # extrapolated with the recorded plate velocity so the
                        # blur stays glued to the true plate position instead
                        # of drifting back to geometry.
                        pcx, pcy, pw, ph = track.ema
                        if dt > 0:
                            vx, vy = track.plate_vel
                            # First clamp: raw velocity has to stay plausible.
                            # Second clamp: drift per frame vs. the last emitted
                            # plate zone (an EMA zone can't just teleport 30px —
                            # that is detection jitter, not plate motion).
                            if (abs(vx) <= self.predict_max_disp
                                    and abs(vy) <= self.predict_max_disp):
                                if track.plate_zone_prev is not None:
                                    dx = (pcx + vx * dt) - track.plate_zone_prev[0]
                                    dy = (pcy + vy * dt) - track.plate_zone_prev[1]
                                    m = max(np.hypot(dx, dy), 1e-6)
                                    if m > self.moto_plate_max_drift:
                                        f = self.moto_plate_max_drift * dt / m
                                        pcx = track.plate_zone_prev[0] + dx * f
                                        pcy = track.plate_zone_prev[1] + dy * f
                                    else:
                                        pcx += vx * dt
                                        pcy += vy * dt
                                else:
                                    pcx += vx * dt
                                    pcy += vy * dt
                        pad = self.moto_plate_pad
                        zone = (max(0, int(pcx - pw * 0.5) - pad),
                                max(0, int(pcy - ph * 0.5) - pad),
                                int(pcx + pw * 0.5) + pad,
                                int(pcy + ph * 0.5) + pad)
                        track.zone_ema_update(zone, self.ema_alpha)
                        ex, ey, ew, eh = track.zone_ema
                        zone = (int(ex - ew * 0.5), int(ey - eh * 0.5),
                                int(ex + ew * 0.5), int(ey + eh * 0.5))
                        track.plate_zone_prev = ((ex), (ey))
                        effective.append((*self._clamp_emit(track, zone), -2.0, "anchor", track_id))
                        continue
                    close = _is_close(track.box)
                    # While the vehicle detector drops the moto, follow it with
                    # the box-velocity prediction instead of freezing in place.
                    # Exception: a close-up moto — pixel velocity is unreliable
                    # at close range, so the zone stays glued to the last box
                    # confirmed by a real detection.
                    base_box = track.box
                    if track.miss_count > 0 and not close:
                        pred = track.predict_box(self.frame_idx,
                                                 self.predict_max_disp)
                        if pred is not None:
                            base_box = pred
                    elif close and track.last_detected_frame < self.frame_idx:
                        base_box = track.last_detected_box
                    if track.zone_ghost_seeded:
                        # First frame back after the track died: emit the ghost
                        # zone itself — no anchor recompute, no EMA blend — so
                        # the blur never jumps when the detector re-acquires
                        # the same motorcycle.
                        ex, ey, ew, eh = track.zone_ema
                        zone = (int(ex - ew * 0.5), int(ey - eh * 0.5),
                                int(ex + ew * 0.5), int(ey + eh * 0.5))
                        zone = _extend_to_edge(zone, base_box)
                        track.zone_ghost_seeded = False
                        effective.append((*self._clamp_emit(track, zone), -2.0, "anchor", track_id))
                        continue
                    bh  = base_box[3] - base_box[1]
                    near = fh > 0 and bh > self.moto_near_frac * fh
                    # The YOLO box of a near moto includes the rider's head —
                    # the plate sits below the box centre, so the zone anchor
                    # slides down as the box grows taller.
                    y_frac = self.moto_anchor_y
                    if near:
                        y_frac = min(self.moto_anchor_y_max,
                                     self.moto_anchor_y + 0.20 * bh / fh)
                    # The zone side must scale with the box: a fixed minimum
                    # (moto_zone_min_side) that fits small/far motos leaves a
                    # near moto's plate half-covered (its plate is ~40-45% of
                    # the YOLO box height). Scale the floor by the box height.
                    min_side = max(self.moto_zone_min_side, int(bh * 0.32))
                    zone = track.anchor_rect(self.moto_anchor_frac,
                                             y_frac,
                                             self.moto_anchor_pad,
                                             base_box=base_box,
                                             min_side=min_side)
                    if near:
                        # Widened zone covers the horizontal offset of a
                        # leaning plate; extended to any frame edge the box
                        # touches.
                        zone = _widen_zone(zone, self.moto_close_zone_w)
                    zone = _extend_to_edge(zone, base_box)
                    track.zone_ema_update(zone, self.ema_alpha)
                    ex, ey, ew, eh = track.zone_ema
                    zone = (int(ex - ew * 0.5), int(ey - eh * 0.5),
                            int(ex + ew * 0.5), int(ey + eh * 0.5))
                    effective.append((*self._clamp_emit(track, zone), -2.0, "anchor", track_id))
                continue

            indexed = [(i, p) for i, p in enumerate(plate_rects)
                       if i not in claimed and _overlaps(p[:4], track.box)]

            if indexed:
                # Pick the history-consistent winner first: it is the box the
                # trajectory is recorded from, and in "kalman" mode also the one
                # measurement the filter is allowed to see.
                if track.plate.has_history and len(track.plate._data) >= _GATE_HISTORY:
                    predicted = track.plate.predict_rect(self.frame_idx,
                                                         self.predict_expand_max)
                    if predicted is not None:
                        gated = [(i, p) for i, p in indexed if _overlaps(p[:4], predicted)]
                        candidates = gated if gated else indexed
                    else:
                        candidates = indexed
                else:
                    candidates = indexed

                best_i, best = max(candidates,
                                   key=lambda ip: ip[1][4] if len(ip[1]) > 4 else 1.0)

                # Add ALL candidates — blur every distinct detection in this box
                for i, p in indexed:
                    if track.cls == 3:   # motorcycle — near-square plate
                        p = (*track.clamp_ar(p[:4], self.moto_ar_min, self.moto_ar_max),
                             *p[4:])
                    conf   = p[4] if len(p) > 4 else 1.0
                    source = p[5] if len(p) > 5 else "sahi"
                    if self.zone_filter == "kalman" and i == best_i:
                        # Emit the filtered estimate instead of the raw box: the
                        # detector's frame-to-frame jitter (amplified by
                        # detect_scale) is what makes the blur wobble.
                        self._kf_observe(track, p[:4], conf)
                        self._kf_emit(effective, track, track_id, 0,
                                      conf, source, measured=p)
                    else:
                        effective.append((*self._clamp_emit(track, p[:4]),
                                          conf, source, track_id))
                    claimed.add(i)
                # EMA-smooth the plate box so the blur doesn't jitter between
                # detection positions frame-to-frame (anti-flicker).
                smoothed = track.ema_update(best[:4], self.ema_alpha)
                ex1, ey1 = int(smoothed[0] - smoothed[2] * 0.5), int(smoothed[1] - smoothed[3] * 0.5)
                ex2, ey2 = int(smoothed[0] + smoothed[2] * 0.5), int(smoothed[1] + smoothed[3] * 0.5)
                best_smoothed = (ex1, ey1, ex2, ey2)
                track.plate.record(self.frame_idx, *best_smoothed,
                                   best[4] if len(best) > 4 else 1.0)
                track.set_plate_anchor(best_smoothed)
                if gray is not None:
                    track.plate.refresh_lk(gray, *best_smoothed)
            else:
                # No detection overlaps this vehicle — advance the miss counter
                track.plate.miss_count += 1
                if (self.zone_filter == "kalman" and track.kf is not None
                        and 1 <= track.plate.miss_count <= self.max_gap_frames):
                    # Gap-fill from the filter: the velocity it already estimated
                    # replaces the anchored/LK/velocity chain, whose three
                    # different extrapolations are what makes the zone jump
                    # between frames. Still clamped inside the vehicle box (the
                    # safety property of predict_anchored) so a long coast can
                    # never wander onto roadside objects.
                    self._kf_advance(track)
                    zone = _clamp_to_box(self._kf_zone(track, 0), track.box)
                    if zone[2] > zone[0] and zone[3] > zone[1]:
                        effective.append((*zone, -1.0, "pred", track_id))
                        continue
                if 1 <= track.plate.miss_count <= self.max_gap_frames \
                        and (track.plate.has_history or track.plate_anchor is not None):
                    # PRIMARY: anchored prediction — apply the last known
                    # plate offset to the *current* vehicle box. ByteTrack
                    # keeps following the vehicle, so the blur stays glued
                    # to it instead of drifting onto background (trees, bins).
                    predicted = track.predict_anchored(self.predict_expand_max)
                    # SECONDARY: LK optical flow refines the anchor position;
                    # velocity extrapolation is the last resort. Both are
                    # sanity-capped so a lost plate blinks rather than flying.
                    if predicted is not None and gray is not None:
                        lk = track.plate.predict_rect_lk(
                            gray, self.predict_expand_max, self.predict_max_disp
                        )
                        if lk is not None:
                            # LK gives pixel-level motion of the plate texture;
                            # blend toward it but stay within the vehicle clamp
                            # by re-running the anchored clamp on the LK box.
                            lx1, ly1, lx2, ly2 = lk
                            vx1, vy1, vx2, vy2 = track.box
                            pad_x = (vx2 - vx1) * 0.15
                            pad_y = (vy2 - vy1) * 0.15
                            clamped = (max(int(vx1 - pad_x), lx1),
                                       max(int(vy1 - pad_y), ly1),
                                       min(int(vx2 + pad_x), lx2),
                                       min(int(vy2 + pad_y), ly2))
                            predicted = clamped
                    elif predicted is None:
                        predicted = track.plate.predict_rect(
                            self.frame_idx, self.predict_expand_max,
                            self.predict_max_disp
                        )
                    if predicted is not None:
                        # conf=-1 marks gap-fill; source 'pred' lets the overlay
                        # render these as dashed yellow boxes instead of solid green
                        effective.append((*self._clamp_emit(track, predicted),
                                          -1.0, "pred", track_id))
                    elif (self.fallback_enabled
                            and track.frames_seen >= self.fallback_min_frames
                            and track.miss_count > self.max_gap_frames):
                        # Privacy fallback: the plate has been lost beyond the
                        # gap window on a stable vehicle — blur the bottom strip
                        # of the vehicle box instead of leaving it exposed.
                        # conf=-2 + source 'fallback' for the debug overlay.
                        effective.append((*self._clamp_emit(track,
                                                            track.fallback_rect()),
                                          -2.0, "fallback", track_id))

        # Standalone plates not claimed by any vehicle track pass through as-is.
        # tid is None: no vehicle track owns them, so nothing correlates them
        # across frames (which is why they cannot be smoothed or gap-filled).
        for i, p in enumerate(plate_rects):
            if i not in claimed:
                conf   = p[4] if len(p) > 4 else 1.0
                source = p[5] if len(p) > 5 else "sahi"
                effective.append((*p[:4], conf, source, None))

        self.frame_idx += 1
        return effective, []   # suppressed is always empty — nothing is discarded
