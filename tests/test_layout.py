import numpy as np
import pytest

from cozmo.geometry.layout import _rectilinear_cleanup, build_layout
from tests.test_rooms import FLOOR, rotation_about_y, two_room_apartment


def layout_of(**kwargs):
    points, walk = two_room_apartment(**kwargs)
    return build_layout(points, FLOOR, walk)


def in_apartment_frame(xz, yaw):
    """World (x, z) back to the apartment's own coordinates."""
    xyz = np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]]) @ rotation_about_y(yaw)
    return xyz[:, [0, 2]]


@pytest.mark.parametrize("yaw", [0.0, 25.0])
def test_two_rectangular_rooms_joined_by_one_door(yaw):
    lay = layout_of(yaw_deg=yaw)
    a, b = lay.outlines[1], lay.outlines[2]
    assert len(a.walls) == 4 and len(b.walls) == 4
    assert sorted(w.length_m for w in a.walls) == pytest.approx([3.0, 3.0, 4.0, 4.0], abs=0.01)
    assert sorted(w.length_m for w in b.walls) == pytest.approx([3.0, 3.0, 3.0, 3.0], abs=0.01)
    assert a.area_m2 == pytest.approx(12.0, abs=0.06) and b.area_m2 == pytest.approx(9.0, abs=0.06)
    assert all(w.support > 0 for w in a.walls + b.walls)

    assert sorted((o.room_id, o.other_room) for o in lay.openings) == [(1, 2), (2, 1)]
    for o in lay.openings:
        assert o.width_m == pytest.approx(0.9, abs=0.03)
        wall = lay.outlines[o.room_id].walls[o.wall_index]
        centre = wall.start + (wall.end - wall.start) / wall.length_m * o.offset_m
        x, z = in_apartment_frame(centre[None], yaw)[0]
        assert x == pytest.approx(4.05, abs=0.06) and z == pytest.approx(1.45, abs=0.03)


def test_a_cabinet_does_not_notch_the_room():
    lay = layout_of(cabinet=True)
    a = lay.outlines[1]
    assert len(a.walls) == 4
    assert a.area_m2 == pytest.approx(12.0, abs=0.06)


def test_a_seen_lintel_does_not_hide_the_door():
    lay = layout_of(lintel=True)
    assert len(lay.outlines) == 2
    assert sorted((o.room_id, o.other_room) for o in lay.openings) == [(1, 2), (2, 1)]


def test_a_hidden_wall_section_is_wall_not_a_door():
    lay = layout_of(hide=(1.5, 2.3))  # 0.8 m of room A's outer wall unseen at every height
    assert len(lay.outlines) == 2
    assert len(lay.outlines[1].walls) == 4
    assert lay.outlines[1].area_m2 == pytest.approx(12.0, abs=0.06)
    assert all(o.other_room is not None for o in lay.openings)  # no door to the outside


def test_an_opening_wider_than_a_door_is_open_plan():
    lay = layout_of(door=(0.3, 2.7))
    assert len(lay.outlines) == 1


def test_a_wall_with_no_points_stays_unsnapped():
    lay = layout_of(drop_wall_z0=True)
    unsnapped = [w for w in lay.outlines[1].walls if w.support == 0]
    assert len(unsnapped) == 1
    mid = (unsnapped[0].start + unsnapped[0].end) / 2
    assert mid[1] == pytest.approx(0.0, abs=0.02)


def test_short_jogs_are_removed():
    square_with_jog = np.array([[0, 0], [4, 0], [4, 1.5], [4.05, 1.5], [4.05, 3], [0, 3]], float)
    cleaned = _rectilinear_cleanup(square_with_jog)
    assert len(cleaned) == 4
