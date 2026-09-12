import numpy as np
import pytest

from cozmo.geometry.fusion import PointSet
from cozmo.geometry.planes import HorizontalPlane
from cozmo.geometry.rooms import build_room_map, room_ceilings

FLOOR = HorizontalPlane(height=0.0, spread=0.005, count=100000)


def two_room_apartment(door=(1.0, 1.9), ceiling_b=None, yaw_deg=0.0):
    """Room A: x 0..4, z 0..3. Room B: x 4.1..7.1, z 0..3. A 10 cm wall between them at x 4.0..4.1
    with a doorway at the given z range. Wall normals face into the room they were seen from, as
    real LiDAR normals face the camera. Optionally a ceiling over room B, and the whole apartment
    rotated about the vertical axis by yaw_deg."""
    rng = np.random.default_rng(1)
    xyz, nrm = [], []

    def add(points, normal):
        xyz.append(points)
        nrm.append(np.tile(normal, (len(points), 1)))

    def floor(x0, x1, z0, z1, per_m2=400):
        n = int((x1 - x0) * (z1 - z0) * per_m2)
        add(np.column_stack([rng.uniform(x0, x1, n), rng.normal(0, 0.005, n), rng.uniform(z0, z1, n)]), [0, 1, 0])

    def wall_at_x(x, z0, z1, facing, per_m=600):
        n = int((z1 - z0) * per_m)
        add(np.column_stack([x + rng.normal(0, 0.005, n), rng.uniform(0.3, 1.8, n), rng.uniform(z0, z1, n)]),
            [facing, 0, 0])

    def wall_at_z(z, x0, x1, facing, per_m=600):
        n = int((x1 - x0) * per_m)
        add(np.column_stack([rng.uniform(x0, x1, n), rng.uniform(0.3, 1.8, n), z + rng.normal(0, 0.005, n)]),
            [0, 0, facing])

    floor(0, 4, 0, 3)
    floor(4.1, 7.1, 0, 3)
    floor(4.0, 4.1, *door)
    for x0, x1 in ((0, 4), (4.1, 7.1)):
        wall_at_z(0, x0, x1, +1)
        wall_at_z(3, x0, x1, -1)
    wall_at_x(0, 0, 3, +1)
    wall_at_x(7.1, 0, 3, -1)
    for x, facing in ((4.0, -1), (4.1, +1)):
        wall_at_x(x, 0, door[0], facing)
        wall_at_x(x, door[1], 3, facing)
    if ceiling_b is not None:
        n = 4000
        add(np.column_stack([rng.uniform(4.1, 7.1, n), ceiling_b + rng.normal(0, 0.005, n), rng.uniform(0, 3, n)]), [0, -1, 0])

    xyz, nrm = np.vstack(xyz), np.vstack(nrm)
    walk = np.column_stack([np.linspace(1, 6, 60), np.full(60, 1.5), np.full(60, 1.45)])
    if yaw_deg:
        t = np.radians(yaw_deg)
        about_y = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        xyz, nrm, walk = xyz @ about_y.T, nrm @ about_y.T, walk @ about_y.T
    points = PointSet(xyz.astype(np.float32), nrm.astype(np.float32), np.zeros(len(xyz), np.int32),
                      np.full(len(xyz), 1.5, np.float32))
    return points, walk


def test_two_rooms_joined_by_one_doorway():
    points, walk = two_room_apartment()
    rooms = build_room_map(points, FLOOR, walk)
    assert rooms.room_ids == [1, 2]
    assert rooms.areas_m2[1] == pytest.approx(12.0, rel=0.1)   # room A, largest first
    assert rooms.areas_m2[2] == pytest.approx(9.0, rel=0.1)
    assert len(rooms.doorways) == 1
    door = rooms.doorways[0]
    assert door.rooms == (1, 2)
    assert door.width_m == pytest.approx(0.9, abs=0.15)
    assert door.center_xz[1] == pytest.approx(1.45, abs=0.1)


def test_points_are_assigned_to_the_right_room():
    points, walk = two_room_apartment()
    rooms = build_room_map(points, FLOOR, walk)
    assert list(rooms.room_of(np.array([[2.0, 1.5], [5.5, 1.5]]))) == [1, 2]


def test_a_wide_opening_makes_one_room():
    points, walk = two_room_apartment(door=(0.3, 2.7))  # 2.4 m opening: open plan
    rooms = build_room_map(points, FLOOR, walk)
    assert rooms.room_ids == [1]


def test_ceiling_is_reported_only_where_it_was_seen():
    points, walk = two_room_apartment(ceiling_b=2.3)
    rooms = build_room_map(points, FLOOR, walk)
    ceilings = room_ceilings(points, FLOOR, rooms)
    assert ceilings[1] is None
    assert ceilings[2].height == pytest.approx(2.3, abs=0.002)
