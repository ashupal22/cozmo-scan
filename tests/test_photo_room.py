"""Photo tier room model on a synthetic room: the box, the unseen side, and a door seen through."""
import numpy as np

from cozmo.geometry.fusion import PointSet
from cozmo.geometry.planes import HorizontalPlane
from cozmo.photo.room import find_doors, fit_box


def _room(door=(1.0, 1.9), rng=np.random.default_rng(0)):
    """4 x 3 m room, walls 2.4 m high, a door in the wall at x = 4 from z = door[0] to door[1], and the next
    room's floor seen through it. Camera in the doorway at x = 0.3."""
    pts, nrm = [], []

    def add(xyz, n):
        pts.append(xyz)
        nrm.append(np.repeat([n], len(xyz), 0))

    g = lambda a, b, n: rng.uniform(a, b, n)  # noqa: E731
    add(np.column_stack([g(0, 4, 4000), np.zeros(4000), g(0, 3, 4000)]), [0, 1, 0])          # floor
    for x, n in ((0.0, [1, 0, 0]), (4.0, [-1, 0, 0])):
        z, y = g(0, 3, 3000), g(0, 2.4, 3000)
        keep = ~((x == 4.0) & (z > door[0]) & (z < door[1]) & (y < 2.0))
        add(np.column_stack([np.full(keep.sum(), x), y[keep], z[keep]]), n)
    for z0, n in ((0.0, [0, 0, 1]), (3.0, [0, 0, -1])):
        add(np.column_stack([g(0, 4, 3000), g(0, 2.4, 3000), np.full(3000, z0)]), n)
    add(np.column_stack([g(4.2, 5.5, 800), g(0.06, 0.6, 800), g(door[0] + 0.05, door[1] - 0.05, 800)]), [0, 1, 0])
    xyz = np.concatenate(pts).astype(np.float32)
    return PointSet(xyz, np.concatenate(nrm).astype(np.float32), np.zeros(len(xyz), np.int32),
                    np.full(len(xyz), 1.4, np.float32))


def test_box_door_and_unseen_side():
    points = _room()
    floor = HorizontalPlane(0.0, 0.005, 4000)
    cameras = np.array([[0.3, 1.4, 1.5], [0.3, 1.4, 1.4]])
    yaw, ext, seen, _, _, cam = fit_box(points, floor, cameras)
    axis_x = 0 if abs(np.cos(np.radians(yaw))) > 0.7 else 1
    span_x = ext[(axis_x, 1)] - ext[(axis_x, -1)]
    span_z = ext[(1 - axis_x, 1)] - ext[(1 - axis_x, -1)]
    assert abs(abs(span_x) - 4.0) < 0.1 and abs(abs(span_z) - 3.0) < 0.1
    assert sum(v is None for v in seen.values()) == 1          # the wall behind the camera
    doors = find_doors(points, floor, yaw, ext, seen)
    assert len(doors) == 1
    assert abs(doors[0][2] - 0.9) <= 0.1
