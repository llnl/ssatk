"""Accepted-time sensing/control and continuous bounded actuator response."""

from copy import deepcopy
import json

import numpy as np

__all__ = ["LaunchActuator", "LaunchControl"]


def _record(value):
    # Reject nonfinite callback data before it reaches retained evidence.
    json.dumps(value, allow_nan=False, default=lambda x: x.tolist())
    return deepcopy(value)


class LaunchActuator:
    """Channels with hard bounds, slew rates and first-order lag.

    Arrays use declared channel units; rate is units/s and lag is s.
    value(t) solves dx/dt = clip((target-x)/lag, +/-rate) without mutation.
    Zero lag gives a slew-limited ramp. No coupling/backlash or actuator
    reaction momentum is inferred; supply a calibrated model when required.
    """

    def __init__(self, *, initial, lower, upper, rate, lag, units, source):
        arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in
                                      (initial, lower, upper, rate, lag)])
        self.initial, self.lower, self.upper, self.rate, self.lag = [v.copy() for v in arrays]
        if (self.initial.ndim != 1 or not self.initial.size
                or not all(np.isfinite(v).all() for v in arrays)
                or np.any(self.lower >= self.upper) or np.any(self.rate <= 0)
                or np.any(self.lag < 0) or np.any(self.initial < self.lower)
                or np.any(self.initial > self.upper) or len(units) != self.initial.size
                or any(not u for u in units) or not source):
            raise ValueError("actuator requires finite bounded channels, positive rates, nonnegative lag, units and source")
        self.units, self.source = tuple(units), source
        self.target, self._value = self.initial.copy(), self.initial.copy()
        self._time = None

    def value(self, time):
        """Actual values at or after the last accepted command epoch."""
        if not np.isfinite(time) or self._time is None or time < self._time:
            raise ValueError("actuator requires an initialized, nondecreasing epoch")
        dt = float(time-self._time)
        distance = np.abs(self.target-self._value)
        ramp_time = np.maximum(0., distance/self.rate-self.lag)
        travel = np.minimum(distance, self.rate*dt)
        delayed = (self.lag > 0) & (dt > ramp_time)
        travel[delayed] = distance[delayed]-np.minimum(distance[delayed], self.rate[delayed]*self.lag[delayed])*np.exp(
            -(dt-ramp_time[delayed])/self.lag[delayed])
        return self._value + np.sign(self.target-self._value)*travel

    def command(self, time, target):
        """Latch at an accepted boundary; return the saturated target."""
        target = np.asarray(target, dtype=float)
        if target.shape != self.initial.shape or not np.isfinite(target).all() or not np.isfinite(time):
            raise ValueError("command must match finite actuator channels and epoch")
        if self._time is not None:
            self._value = self.value(time)
        self._time = float(time)
        self.target = np.clip(target, self.lower, self.upper)
        return self.target.copy()


class LaunchControl:
    """One-run sensor -> estimator -> guidance -> controller pipeline.

    sensor(copied_truth) returns recordable measurements; bias/noise/dropout
    belong to the selected sensor. estimator(time, delivered, previous_estimate)
    receives only packets with sample/delivery epochs. guidance(time, estimate)
    returns a reference; controller(time, estimate, reference) returns actuator
    channels. These interfaces never give estimator/guidance/controller truth.
    Supply a prior estimate; unavailable measurements never substitute truth.
    Callbacks return finite JSON-recordable data, including declared covariance.

    Physical events precede updates at equal epochs. Acquisition precedes
    delivery/estimation, guidance/control and activation, in that order.
    Periods and latencies are seconds, independent of output and solver steps.
    Actuator response is continuous, not a controller evaluation in the RHS.
    """

    def __init__(self, *, sensor, estimator, guidance, controller, actuator,
                 estimate, sensor_period, control_period, sensor_latency=0., command_latency=0.):
        if (not all(callable(f) for f in (sensor, estimator, guidance, controller))
                or not isinstance(actuator, LaunchActuator)
                or not np.isfinite([sensor_period, control_period, sensor_latency, command_latency]).all()
                or min(sensor_period, control_period) <= 0 or min(sensor_latency, command_latency) < 0):
            raise ValueError("control requires callbacks, actuator, positive periods and nonnegative latencies")
        self.sensor, self.estimator, self.guidance, self.controller = sensor, estimator, guidance, controller
        self.actuator, self.estimate = actuator, _record(estimate)
        self.sensor_period, self.control_period = float(sensor_period), float(control_period)
        self.sensor_latency, self.command_latency = float(sensor_latency), float(command_latency)
        self.history, self._measurements, self._commands = [], [], []
        self._epoch = None
        self._sensor_index = self._control_index = 0
        self.reference = None
        self.requested = actuator.initial.copy()

    def start(self, epoch):
        """Bind once to the exact initial GPS epoch of propagate_launch."""
        if self._epoch is not None or self.actuator._time is not None or not np.isfinite(epoch):
            raise ValueError("use a fresh control and actuator for each run")
        self._epoch = float(epoch)
        if min(self.sensor_period, self.control_period) <= 8*np.spacing(abs(epoch)):
            raise ValueError("sample periods must be resolvable at this GPS epoch")
        self.actuator.command(epoch, self.actuator.initial)

    @property
    def next_time(self):
        """Next acquisition, control, measurement delivery or command epoch."""
        if self._epoch is None:
            raise ValueError("control has not started")
        return min(self._epoch+self._sensor_index*self.sensor_period,
                   self._epoch+self._control_index*self.control_period,
                   *(p["delivery_time_gps_s"] for p in self._measurements),
                   *(t for t, _ in self._commands))

    def update(self, truth, phase):
        """Run one exact accepted boundary, called by the launch event runner."""
        time = truth.t
        if time != self.next_time:
            raise ValueError("updates must occur at their exact scheduled epoch")
        acquired = []
        if time == self._epoch+self._sensor_index*self.sensor_period:
            packet = {"sample_time_gps_s": time, "delivery_time_gps_s": time+self.sensor_latency,
                      "measurement": _record(self.sensor(deepcopy(truth)))}
            self._measurements.append(packet)
            acquired.append(deepcopy(packet))
            self._sensor_index += 1
        delivered = [p for p in self._measurements if p["delivery_time_gps_s"] <= time]
        if delivered:
            self.estimate = _record(self.estimator(time, deepcopy(delivered), deepcopy(self.estimate)))
            self._measurements = [p for p in self._measurements if p["delivery_time_gps_s"] > time]
        control_due = time == self._epoch+self._control_index*self.control_period
        if control_due:
            self.reference = _record(self.guidance(time, deepcopy(self.estimate)))
            command = self.controller(time, deepcopy(self.estimate), deepcopy(self.reference))
            self.requested = np.asarray(command, dtype=float).copy()
            if self.requested.shape != self.actuator.initial.shape or not np.isfinite(self.requested).all():
                raise ValueError("controller output must match finite actuator channels")
            self._commands.append((time+self.command_latency, self.requested.copy()))
            self._control_index += 1
        activated = []
        while self._commands and self._commands[0][0] <= time:
            epoch, command = self._commands.pop(0)
            activated.append({"time_gps_s": epoch, "requested": command,
                              "bounded": self.actuator.command(epoch, command)})
        self.history.append({"time_gps_s": time, "phase": phase, "acquired": acquired,
                             "delivered": deepcopy(delivered), "estimate": deepcopy(self.estimate),
                             "reference": deepcopy(self.reference), "control_due": control_due,
                             "requested": self.requested.copy(), "activated": activated,
                             "actuator_target": self.actuator.target.copy(), "actual": self.actuator.value(time)})
        if self.next_time <= time:
            raise ValueError("schedule cannot advance at this GPS precision")
