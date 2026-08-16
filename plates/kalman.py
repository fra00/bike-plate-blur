# ─── Constant-velocity box filter ──────────────────────────────────────────
"""
One Kalman filter per tracked plate zone, replacing the cascade of two EMAs.

Why a filter instead of exponential averages: an EMA can only trade smoothness
for lag. On a target moving at constant speed it settles at a fixed distance
behind it — v(1-alpha)/alpha px, doubled when two EMAs are chained — which is
why the blur trails the plate. And because the previous design only added the
velocity term on frames *without* a detection, the zone alternated between
lagging and over-shooting at the detection burst rate: the visible wobble.

A constant-velocity Kalman filter fixes both at once. Velocity is a state with
its own uncertainty rather than a noisy difference of already-smoothed centres,
so it is applied on *every* frame (no alternation), and a gap is just prediction
without correction. The growing covariance during a gap is a principled measure
of how far the plate may have moved, so the padding can grow with it instead of
using a fixed "2 px per missed frame" rule.

State: [cx, cy, w, h, vx, vy, vw, vh] — position, size and their velocities, in
pixels and pixels/frame.
"""
import math

import numpy as np


_DIM = 8


class BoxFilter:
    """Constant-velocity Kalman filter over a box's centre and size.

    process_pos / process_size are the per-frame process-noise standard
    deviations (px): how much genuine acceleration the plate may have between
    frames. Higher = follows sharp manoeuvres, lower = smoother.
    meas_pos / meas_size are the measurement standard deviations (px) for a
    full-confidence detection; they are scaled up for weak detections, so a
    0.15-confidence box nudges the estimate instead of yanking it.
    """

    def __init__(self, box, conf: float = 1.0,
                 process_pos: float = 3.0, process_size: float = 1.5,
                 meas_pos: float = 3.0, meas_size: float = 3.0,
                 max_meas_scale: float = 6.0,
                 vel_decay: float = 0.85, vel_decay_after: int = 3):
        cx, cy, w, h = _to_cxcywh(box)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Position/size start at measurement accuracy; velocity is unknown, so
        # its variance is large enough for the first two frames to define it.
        self.P = np.diag([meas_pos ** 2, meas_pos ** 2,
                          meas_size ** 2, meas_size ** 2,
                          (meas_pos * 4) ** 2, (meas_pos * 4) ** 2,
                          (meas_size * 4) ** 2, (meas_size * 4) ** 2])
        self.process_pos    = float(process_pos)
        self.process_size   = float(process_size)
        self.meas_pos       = float(meas_pos)
        self.meas_size      = float(meas_size)
        self.max_meas_scale = float(max_meas_scale)
        # Constant velocity is only credible for as long as the plate keeps being
        # seen. Extrapolating a stale velocity over a long gap walks the estimate
        # arbitrarily far from the vehicle, so after vel_decay_after unobserved
        # frames the velocity decays geometrically: the estimate coasts, slows and
        # settles instead of flying off.
        self.vel_decay       = float(vel_decay)
        self.vel_decay_after = int(vel_decay_after)
        self.age            = 1     # frames since the filter was created
        self.since_update   = 0     # frames since the last real detection
        self._last_conf     = float(conf)

    # ── prediction / correction ───────────────────────────────────────────────

    def predict(self, dt: int = 1) -> None:
        """Advance the state by *dt* frames, one frame at a time."""
        for _ in range(max(0, int(dt))):
            self._predict_one()

    def _predict_one(self) -> None:
        if self.since_update >= self.vel_decay_after:
            self.x[4:] *= self.vel_decay
        F = np.eye(_DIM)
        for i in range(4):
            F[i, i + 4] = 1.0
        self.x = F @ self.x
        # Size can never go negative, however long the gap.
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        q_pos  = self.process_pos  ** 2
        q_size = self.process_size ** 2
        Q = np.diag([q_pos, q_pos, q_size, q_size,
                     q_pos, q_pos, q_size, q_size])
        self.P = F @ self.P @ F.T + Q
        self.age          += 1
        self.since_update += 1

    def update(self, box, conf: float = 1.0, noise_scale: float = 1.0) -> None:
        """Correct the state with a detected *box* of confidence *conf*.

        *noise_scale* multiplies the measurement standard deviation on top of
        the confidence scaling. Use it for geometric (non-detector) updates —
        an anchor guessed from the vehicle box is typically ~5× noisier than a
        real plate detection — so the estimate is nudged, not yanked.
        """
        z = np.array(_to_cxcywh(box), dtype=np.float64)
        H = np.zeros((4, _DIM))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        R = np.diag(self._meas_var(conf, noise_scale))
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(_DIM) - K @ H) @ self.P
        self.since_update = 0
        self._last_conf   = float(conf)

    def _meas_var(self, conf: float, noise_scale: float = 1.0) -> list:
        """Measurement variances, inflated for low-confidence / weak sources.

        A detection at the 0.15 audit floor is positionally much less reliable
        than one at 0.9; scaling by 1/conf (capped) expresses that instead of
        treating every box as equally trustworthy. *noise_scale* is an extra
        multiplier for non-detector measurements (geometric anchors).
        """
        scale = min(self.max_meas_scale, 1.0 / max(conf, 1e-3))
        scale *= max(1.0, float(noise_scale))
        return [(self.meas_pos * scale) ** 2, (self.meas_pos * scale) ** 2,
                (self.meas_size * scale) ** 2, (self.meas_size * scale) ** 2]

    # ── queries ───────────────────────────────────────────────────────────────

    def box(self, pad: float = 0.0) -> tuple:
        """Current estimate as (x1, y1, x2, y2), expanded by *pad* px."""
        cx, cy, w, h = self.x[:4]
        w = max(1.0, w) + 2 * pad
        h = max(1.0, h) + 2 * pad
        return (int(round(cx - w * 0.5)), int(round(cy - h * 0.5)),
                int(round(cx + w * 0.5)), int(round(cy + h * 0.5)))

    @property
    def centre(self) -> tuple:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple:
        return float(self.x[4]), float(self.x[5])

    def position_sigma(self) -> float:
        """Standard deviation of the position estimate (px).

        Grows while the plate is missing, which is exactly the amount of extra
        padding needed to keep covering it through a gap.
        """
        return math.sqrt(max(0.0, float(self.P[0, 0] + self.P[1, 1]) * 0.5))

    def gate_distance(self, box) -> float:
        """Mahalanobis distance of *box* from the predicted position.

        Replaces the fixed px/frame displacement caps: a detection is judged
        against how uncertain the estimate currently is, so a well-established
        track rejects a jump that a freshly-born one still accepts.
        """
        z = np.array(_to_cxcywh(box)[:2], dtype=np.float64)
        d = z - self.x[:2]
        S = self.P[:2, :2] + np.diag(self._meas_var(1.0)[:2])
        try:
            return float(math.sqrt(max(0.0, d @ np.linalg.inv(S) @ d)))
        except np.linalg.LinAlgError:
            return float("inf")


def _to_cxcywh(box) -> tuple:
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5,
            max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1)))
