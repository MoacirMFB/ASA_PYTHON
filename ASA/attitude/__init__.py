"""Spacecraft attitude determination, kinematics, dynamics, and control.

Python port of ``AttitudeDeterminationLibrary.m`` from the ASA_MATLAB repo,
split by topic:

- :mod:`~ASA.attitude.dcm`         direction cosine matrices, Euler axis/angle, Euler sequences
- :mod:`~ASA.attitude.quaternions` quaternion algebra and quaternion/DCM conversions
- :mod:`~ASA.attitude.kinematics`  DCM, Euler-angle, and quaternion rates
- :mod:`~ASA.attitude.rigidbody`   Euler's equations, gravity moments, inertia properties
- :mod:`~ASA.attitude.estimation`  TRIAD, QUEST, Davenport's q-method, Wahba covariances
- :mod:`~ASA.attitude.sensors`     gyro, star tracker, and center-of-light models
- :mod:`~ASA.attitude.wheels`      reaction-wheel configuration, control, torque distribution

Conventions: quaternions are scalar-last, ``q = [q1, q2, q3, q4]``; a
``convention`` flag of ``"col"`` means the DCM acts on column vectors
(``v_B = [C] v_R``) and ``"row"`` means it acts on row vectors.

The MATLAB plotting helpers are deliberately not ported, since ASA depends only
on numpy and scipy.
"""

from . import dcm, estimation, kinematics, quaternions, rigidbody, sensors, wheels
from .dcm import *  # noqa: F401,F403
from .estimation import *  # noqa: F401,F403
from .kinematics import *  # noqa: F401,F403
from .quaternions import *  # noqa: F401,F403
from .rigidbody import *  # noqa: F401,F403
from .sensors import *  # noqa: F401,F403
from .wheels import *  # noqa: F401,F403

__all__ = ["dcm", "estimation", "kinematics", "quaternions", "rigidbody", "sensors", "wheels"]
for _module in (dcm, quaternions, kinematics, rigidbody, estimation, sensors, wheels):
    __all__.extend(_module.__all__)
