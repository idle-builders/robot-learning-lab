"""Timed register model exercising the actual C IMU driver against MuJoCo samples.

This is an explicit sample/hold and I2C timing hypothesis, not an ST silicon model.
Physics supplies past observations only; a bus poll is executed only when its
maximum completion time is in the available history. No future truth is read.
"""

from bisect import bisect_right
import ctypes as ct
import math
import json
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .config import Strict
from .imu import load_imu
from .io import IMUSample


class SensorModel(Strict):
    mode: Literal["ideal", "lsm6dso"] = "ideal"
    sensor_to_torso_rpy_rad: tuple[float, float, float] | None = None
    calibration_to_torso_rpy_rad: tuple[float, float, float] | None = None
    gyro_phase_us: int = Field(default=0, ge=0, le=4807)
    poll_block_us: int = Field(default=0, ge=0, le=100000)
    poll_block_period_us: int = Field(default=20000, ge=2000, le=200000)
    fault: Literal["none", "io", "dropout", "saturation"] = "none"
    fault_start_s: float = Field(default=3, ge=-0.25, le=60)
    fault_duration_s: float = Field(default=0.1, gt=0, le=60)

    @model_validator(mode="after")
    def intervals(self):
        if self.poll_block_us >= self.poll_block_period_us:
            raise ValueError("poll block must leave time to poll")
        if not all(
            math.isfinite(v)
            for v in (
                *(self.sensor_to_torso_rpy_rad or ()),
                *(self.calibration_to_torso_rpy_rad or ()),
            )
        ):
            raise ValueError("finite sensor rotations required")
        return self


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


class NativeSample(ct.Structure):
    _fields_ = [
        ("gyro", ct.c_double * 3),
        ("accel", ct.c_double * 3),
        ("poll_start_us", ct.c_uint64),
        ("earliest_update_us", ct.c_uint64),
        ("sequence", ct.c_uint32),
        ("saturated", ct.c_uint8),
    ]


class NativeState(ct.Structure):
    _fields_ = [
        ("status", ct.c_uint32),
        ("error", ct.c_uint32),
        ("last_poll_us", ct.c_uint64),
        ("deadline_us", ct.c_uint64),
        ("max_poll_gap_us", ct.c_uint64),
        ("last_drain_us", ct.c_uint64),
        ("has_sample", ct.c_uint8),
        ("sample", NativeSample),
    ]


Callback = ct.CFUNCTYPE(
    ct.c_int, ct.c_void_p, ct.c_uint8, ct.POINTER(ct.c_uint8), ct.c_size_t
)


class NativeBus(ct.Structure):
    _fields_ = [("context", ct.c_void_p), ("read", Callback), ("write", Callback)]


class DriverIMU:
    """Owns C state plus six independently held words and bounded virtual bus time."""

    release_us = 250000  # explicit held startup fixture before free-base release

    def __init__(
        self,
        library,
        settings=None,
        *,
        seed=0,
        gyro_bias=0,
        gyro_noise=0,
        accel_noise=0,
        config_path=Path("electronics/imu.yaml"),
        robot=None,
    ):
        self.settings = settings or SensorModel(mode="lsm6dso")
        self.config = load_imu(config_path, robot=robot)
        self.config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()
        self.lib = ct.CDLL(str(library))
        self.lib.imu_init.argtypes = [ct.POINTER(NativeState), ct.c_uint64]
        self.lib.imu_init.restype = None
        self.lib.imu_poll.argtypes = [
            ct.POINTER(NativeState),
            ct.POINTER(NativeBus),
            ct.c_uint64,
        ]
        self.lib.imu_poll.restype = ct.c_int
        self.lib.imu_valid.argtypes = [
            ct.POINTER(NativeState),
            ct.c_uint64,
            ct.c_uint64,
        ]
        self.lib.imu_valid.restype = ct.c_int
        self.state = NativeState()
        self.lib.imu_init(ct.byref(self.state), 0)
        self.read_callback, self.write_callback = (
            Callback(self._read),
            Callback(self._write),
        )
        self.bus = NativeBus(None, self.read_callback, self.write_callback)
        manifest_rotation = np.asarray(
            self.config["sensor_frame"]["torso_from_sensor_rotation"], dtype=float
        )
        self.true_rotation = (
            rotation(self.settings.sensor_to_torso_rpy_rad)
            if self.settings.sensor_to_torso_rpy_rad is not None
            else manifest_rotation
        )
        self.calibration_rotation = (
            rotation(self.settings.calibration_to_torso_rpy_rad)
            if self.settings.calibration_to_torso_rpy_rad is not None
            else manifest_rotation
        )
        self.rng = np.random.default_rng(seed)
        self.gyro_bias, self.gyro_noise, self.accel_noise = (
            gyro_bias,
            gyro_noise,
            accel_noise,
        )
        self.reg = bytearray(256)
        self.reg[0x0F], self.reg[0x12], self.reg[0x18] = 0x6C, 4, 0xE0
        self.raw = np.zeros(6, dtype=np.int16)
        self.unread = [False] * 6
        self.word_time = [0] * 6
        self.word_truth_time = [0] * 6
        self.next_conversion = [None, None]
        self.reset_until = None
        self.cursor_us = 0
        self.next_poll_us = 0
        self.now_us = 0
        self.byte_us = 9e6 / self.config["bus_hz"]
        self.history_times, self.history_values = [], []
        self.polls = []
        self.dropped_words = 0
        self.conversions = [0, 0]
        self.read_end_us = 0
        self.last_word_times, self.last_truth_times = [], []
        self.callback_error = None

    def _fault_at(self, time_us, kind):
        start = self.release_us + round(self.settings.fault_start_s * 1e6)
        end = start + round(self.settings.fault_duration_s * 1e6)
        return self.settings.fault == kind and start <= time_us < end

    def _advance_device(self, to_us):
        if to_us < self.cursor_us - 1e-6:
            raise ValueError("virtual I2C clock regressed")
        if self.reset_until is not None and to_us >= self.reset_until:
            self.reg[0x12] = 4
            self.reset_until = None
        while True:
            available = [
                (t, g)
                for g, t in enumerate(self.next_conversion)
                if t is not None and t <= to_us
            ]
            if not available:
                break
            event, group = min(available)
            self.next_conversion[group] += 1e6 / self.config["odr_hz"]
            if self._fault_at(event, "dropout"):
                continue
            i = bisect_right(self.history_times, event) - 1
            if i < 0:
                raise ValueError("sensor conversion has no past source observation")
            gyro, accel = self.history_values[i]
            values = self.true_rotation.T @ (gyro if group == 0 else accel)
            std = self.gyro_noise if group == 0 else self.accel_noise
            values = values + self.rng.normal(0, std, 3)
            if group == 0:
                values[1] += self.gyro_bias
                if self._fault_at(event, "saturation"):
                    values[1] = 100.0
            scale = 0.0175 * math.pi / 180 if group == 0 else 0.000122 * 9.80665
            counts = np.clip(np.rint(values / scale), -32768, 32767).astype(np.int16)
            self.conversions[group] += 1
            for axis in range(3):
                word = group * 3 + axis
                if self.unread[word]:
                    self.dropped_words += 1
                else:
                    self.raw[word] = counts[axis]
                    self.word_time[word] = round(event)
                    self.word_truth_time[word] = self.history_times[i]
                    self.unread[word] = True
        self.cursor_us = to_us

    def _timeout(self, completion):
        start = self.release_us + round(self.settings.fault_start_s * 1e6)
        end = start + round(self.settings.fault_duration_s * 1e6)
        if self.settings.fault == "io" and self.cursor_us < end and completion >= start:
            self._advance_device(self.cursor_us + self.config["transfer_timeout_us"])
            return True
        return False

    def _read(self, context, reg, data, count):
        try:
            if self._timeout(self.cursor_us + (3 + count) * self.byte_us):
                return 0
            self._advance_device(self.cursor_us + 3 * self.byte_us)
            if reg == 0x22 and count == 12:
                self.last_word_times, self.last_truth_times = [], []
            for i in range(count):
                self._advance_device(self.cursor_us + self.byte_us)
                address = reg + i
                if address == 0x1E:
                    data[i] = int(any(self.unread[:3])) * 2 + int(any(self.unread[3:]))
                elif 0x22 <= address < 0x2E:
                    word, byte = divmod(address - 0x22, 2)
                    data[i] = (int(self.raw[word]) >> (8 * byte)) & 255
                    if byte == 1:
                        self.last_word_times.append(self.word_time[word])
                        self.last_truth_times.append(self.word_truth_time[word])
                        self.unread[word] = False
                else:
                    data[i] = self.reg[address]
            return 1
        except Exception as exc:
            self.callback_error = str(exc)
            return 0

    def _write(self, context, reg, data, count):
        try:
            if self._timeout(self.cursor_us + (2 + count) * self.byte_us):
                return 0
            self._advance_device(self.cursor_us + (2 + count) * self.byte_us)
            for i in range(count):
                address = reg + i
                self.reg[address] = data[i]
                if address == 0x12 and data[i] & 1:
                    self.reset_until = self.cursor_us + 1000  # modeled reset completion
                    self.next_conversion = [None, None]
                    self.unread = [False] * 6
                if address in (0x10, 0x11):
                    group = int(address == 0x10)  # 0 gyro, 1 accelerometer
                    phase = self.settings.gyro_phase_us if group == 0 else 0
                    self.next_conversion[group] = (
                        self.cursor_us + phase + 1e6 / self.config["odr_hz"]
                    )
            return 1
        except Exception as exc:
            self.callback_error = str(exc)
            return 0

    def _budget(self, start):
        status = self.state.status
        if status in (0, 2) and start < self.state.deadline_us or status == 4:
            return 0
        normal = {
            0: 7 * self.byte_us,
            1: 46 * self.byte_us,
            2: 15 * self.byte_us,
            3: 19 * self.byte_us,
        }[status]
        # Startup/configuration can also time out. Reserve the entire normal
        # transaction chain plus one bounded failing callback before executing.
        return normal + (
            self.config["transfer_timeout_us"] if self.settings.fault == "io" else 0
        )

    def advance(self, now_us, truth=None):
        if now_us < self.now_us:
            raise ValueError("sensor model clock regressed")
        if truth is not None:
            t = round(truth.time_s * 1e6)
            if t > now_us or (self.history_times and t < self.history_times[-1]):
                raise ValueError("future or regressed truth source")
            if not self.history_times or t > self.history_times[-1]:
                self.history_times.append(t)
                self.history_values.append(
                    (
                        np.asarray(truth.gyro_rad_s),
                        np.asarray(truth.specific_force_m_s2),
                    )
                )
        self.now_us = now_us
        while True:
            start = self.next_poll_us
            if start >= self.release_us and self.settings.poll_block_us:
                phase = (start - self.release_us) % self.settings.poll_block_period_us
                if phase < self.settings.poll_block_us:
                    start += self.settings.poll_block_us - phase
            if start + self._budget(start) > now_us:
                break
            self._advance_device(start)
            result = self.lib.imu_poll(ct.byref(self.state), ct.byref(self.bus), start)
            if self.callback_error:
                raise ValueError(self.callback_error)
            if self.cursor_us > now_us + 1e-6:
                raise ValueError("driver consumed a future bus completion")
            if result == 1:
                self.read_end_us = round(self.cursor_us)
                self.polls.append(
                    {
                        "sequence": self.state.sample.sequence,
                        "poll_start_us": start,
                        "earliest_update_us": self.state.sample.earliest_update_us,
                        "read_end_us": self.read_end_us,
                        "word_conversion_us": self.last_word_times.copy(),
                        "truth_source_us": self.last_truth_times.copy(),
                        "saturated": bool(self.state.sample.saturated),
                    }
                )
            self.next_poll_us = max(
                start + self.config["poll_interval_us"], math.ceil(self.cursor_us)
            )
            if self.state.status == 4:
                break
        return self.health()

    def warmup(self, specific_force):
        for now in range(0, self.release_us + 1, 2000):
            self.advance(now, IMUSample(now / 1e6, (0, 0, 0), tuple(specific_force)))
        return self.health()

    def feed(self, truth, now_s):
        source = IMUSample(
            truth.time_s + self.release_us / 1e6,
            truth.gyro_rad_s,
            truth.specific_force_m_s2,
        )
        return self.advance(self.release_us + round(now_s * 1e6), source)

    def health(self):
        return {
            "status": self.state.status,
            "error": self.state.error,
            "valid": bool(
                self.lib.imu_valid(
                    ct.byref(self.state), self.now_us, self.config["freshness_us"]
                )
            ),
            "sequence": self.state.sample.sequence,
            "poll_age_us": self.now_us - self.state.sample.poll_start_us
            if self.state.has_sample
            else None,
            "update_age_bound_us": self.now_us - self.state.sample.earliest_update_us
            if self.state.has_sample
            else None,
            "saturated": bool(self.state.sample.saturated),
            "read_end_us": self.read_end_us,
            "max_poll_gap_us": self.state.max_poll_gap_us,
        }

    def observation(self):
        if not self.health()["valid"]:
            raise ValueError("invalid physical-driver IMU observation")
        return IMUSample(
            (self.state.sample.poll_start_us - self.release_us) / 1e6,
            tuple(self.calibration_rotation @ np.asarray(self.state.sample.gyro)),
            tuple(self.calibration_rotation @ np.asarray(self.state.sample.accel)),
        )

    def summary(self):
        post = [r for r in self.polls if r["poll_start_us"] >= self.release_us]
        return {
            "driver": "firmware/imu.c",
            "health": self.health(),
            "held_startup_us": self.release_us,
            "accepted_samples_after_release": len(post),
            "conversions": self.conversions,
            "bdu_discarded_word_updates": self.dropped_words,
            "max_word_conversion_skew_us": max(
                (
                    max(r["word_conversion_us"]) - min(r["word_conversion_us"])
                    for r in post
                ),
                default=0,
            ),
            "max_source_hold_us": max(
                (
                    c - t
                    for r in post
                    for c, t in zip(r["word_conversion_us"], r["truth_source_us"])
                ),
                default=0,
            ),
            "config": self.config,
            "imu_config_sha256": self.config_hash,
            "rotations": {
                "sensor_to_torso": self.true_rotation.tolist(),
                "calibration_to_torso": self.calibration_rotation.tolist(),
                "sensor_source": "experiment_override"
                if self.settings.sensor_to_torso_rpy_rad is not None
                else "imu_manifest",
                "calibration_source": "experiment_override"
                if self.settings.calibration_to_torso_rpy_rad is not None
                else "imu_manifest",
            },
            "model": self.settings.model_dump(),
            "limitations": [
                "sample/hold source at physics rate; no analog sensor/filter/temperature model",
                "word-level BDU model and byte-timed reads are explicit hypotheses, not silicon qualification",
                "9 clocks per byte; START/STOP, clock stretch and physical transfer timing not measured",
                "held startup fixture and modeled 1 ms reset; mounting rotations assumed",
                "poll gaps are scenario inputs, not measured Pico scheduling",
            ],
        }
