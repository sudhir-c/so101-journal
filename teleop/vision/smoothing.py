"""Temporal smoothing for the computed joint values.

``compute_arm_state`` is deliberately pure and stateless.  Smoothing is inherently
stateful, so it lives here and the caller composes the two.  That keeps the math
module trivially reusable and testable.

We use a **One-Euro filter** rather than a fixed exponential moving average.  An
EMA forces a single bad tradeoff: smooth enough to kill jitter when you hold
still, and it visibly lags when you move.  The One-Euro filter adapts its cutoff
frequency to the observed speed of the signal -- heavy smoothing when the value is
nearly static, light smoothing when it is moving fast -- which is exactly the
behaviour you want for live joint readouts.

Reference: Casiez, Roussel & Vogel, "1 Euro Filter" (CHI 2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .pose_math import ArmState

__all__ = ["OneEuroFilter", "ArmStateSmoother"]


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a first-order low-pass at ``cutoff`` Hz."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Scalar One-Euro filter.

    Parameters
    ----------
    min_cutoff:
        Cutoff (Hz) as the signal approaches standstill.  **Lower = steadier**
        when you hold a pose, at the cost of a little lag. This is the main knob.
    beta:
        How aggressively the cutoff opens up with speed.  **Higher = less lag**
        on fast motion, at the cost of letting more jitter through.
    d_cutoff:
        Cutoff for the internal derivative estimate; rarely needs changing.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float | None, t: float) -> float | None:
        """Filter sample ``x`` taken at timestamp ``t`` (seconds).

        ``None`` passes straight through and *resets* the filter, so that when
        tracking drops out and later recovers we snap to the new value instead of
        gliding across the gap from a stale one.
        """
        if x is None:
            self.reset()
            return None

        x = float(x)

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            self._dx_prev = 0.0
            return x

        dt = t - self._t_prev
        if dt <= 0:  # duplicate//out-of-order frame: hold previous estimate
            return self._x_prev
        self._t_prev = t

        # Low-pass the derivative, then let its magnitude drive the cutoff.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


@dataclass
class _Tuning:
    min_cutoff: float
    beta: float


class ArmStateSmoother:
    """Applies a One-Euro filter to each of the four values of an ``ArmState``.

    The per-signal tuning below reflects how noisy each measurement actually is:

    * ``shoulder_lift`` / ``elbow_flex`` ride on large, well-localized pose
      landmarks and are already fairly clean, so they get a light touch.
    * ``wrist_flex`` depends on small hand landmarks and a short forearm-to-hand
      lever arm, so a few pixels of landmark jitter swing it by degrees.  It gets
      the most aggressive smoothing.
    * ``gripper`` drives a physical open/close decision, so steadiness beats
      responsiveness -- a flickering value is worse than a slightly late one.
    """

    DEFAULTS: dict[str, _Tuning] = {
        "shoulder_lift": _Tuning(min_cutoff=1.2, beta=0.06),
        "elbow_flex": _Tuning(min_cutoff=1.2, beta=0.06),
        "wrist_flex": _Tuning(min_cutoff=0.6, beta=0.03),  # noisiest -> steadiest
        "gripper": _Tuning(min_cutoff=0.8, beta=0.04),
    }

    def __init__(self, tuning: dict[str, _Tuning] | None = None):
        cfg = tuning or self.DEFAULTS
        self._filters = {
            name: OneEuroFilter(min_cutoff=t.min_cutoff, beta=t.beta)
            for name, t in cfg.items()
        }

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()

    def __call__(self, state: ArmState, t: float) -> ArmState:
        """Return ``state`` with its four values smoothed. Mutates and returns it."""
        for name, filt in self._filters.items():
            setattr(state, name, filt(getattr(state, name), t))
        return state
