import numpy as np
import pytest

from cozmo.geometry.rooms import build_room_map
from cozmo.geometry.walls import dominant_yaw_deg, outline_rooms, wall_mask
from tests.test_rooms import FLOOR, rotation_about_y, two_room_apartment


@pytest.mark.parametrize("yaw", [0.0, 25.0])
def test_dominant_wall_direction(yaw):
    points, _ = two_room_apartment(yaw_deg=yaw)
    walls = points.subset(wall_mask(points, FLOOR))
    got = dominant_yaw_deg(walls.normal[:, [0, 2]])
    # rotating the scene by +yaw about y turns a +x normal to atan2(z, x) = -yaw
    expected = (-yaw) % 90
    assert min(abs(got - expected), 90 - abs(got - expected)) < 0.3


@pytest.mark.parametrize("yaw", [0.0, 25.0])
def test_outlines_snap_to_wall_faces(yaw):
    points, walk = two_room_apartment(yaw_deg=yaw)
    rooms = build_room_map(points, FLOOR, walk)
    outlines = outline_rooms(rooms, points, FLOOR)
    a, b = outlines[1], outlines[2]  # room A (12 m2) then room B (9 m2)

    assert len(a.vertices) == 4 and len(b.vertices) == 4
    assert sorted(w.length_m for w in a.walls) == pytest.approx([3.0, 3.0, 4.0, 4.0], abs=0.02)
    assert sorted(w.length_m for w in b.walls) == pytest.approx([3.0, 3.0, 3.0, 3.0], abs=0.02)
    assert a.area_m2 == pytest.approx(12.0, abs=0.15)
    assert b.area_m2 == pytest.approx(9.0, abs=0.15)
    assert all(w.support >= 50 for w in a.walls + b.walls)
    assert all(w.face_spread_m < 0.01 for w in a.walls + b.walls)


@pytest.mark.parametrize("yaw", [0.0, 25.0])
def test_wall_evidence_is_attached_to_the_right_wall(yaw):
    # Room A's z=0 wall has no wall points, so exactly one wall must report support 0,
    # and it must be the wall that lies along z=0 (in the apartment's own coordinates).
    points, walk = two_room_apartment(yaw_deg=yaw, drop_wall_z0=True)
    outline = outline_rooms(build_room_map(points, FLOOR, walk), points, FLOOR)[1]
    unsnapped = [w for w in outline.walls if w.support == 0]
    assert len(unsnapped) == 1
    mid = (unsnapped[0].start + unsnapped[0].end) / 2
    back = rotation_about_y(yaw).T @ np.array([mid[0], 0.0, mid[1]])
    assert back[2] == pytest.approx(0.0, abs=0.1)   # on the z=0 wall
    assert 1.0 < back[0] < 3.0                       # somewhere along its length
    assert all(w.support > 0 for w in outline.walls if w is not unsnapped[0])


def test_outline_is_counter_clockwise_and_walls_chain():
    points, walk = two_room_apartment()
    outline = outline_rooms(build_room_map(points, FLOOR, walk), points, FLOOR)[1]
    x, z = outline.vertices[:, 0], outline.vertices[:, 1]
    assert np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)) > 0
    for k, wall in enumerate(outline.walls):
        assert np.allclose(wall.end, outline.walls[(k + 1) % len(outline.walls)].start)
