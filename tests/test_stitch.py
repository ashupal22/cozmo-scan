"""Door-matching stitch solver (cozmo/stitch/solver.py) on small homes where the answer is known exactly."""
import numpy as np
import pytest

from cozmo.stitch.solver import Door, Placement, Room, rot, stitch

T = 0.15   # wall thickness used to build the homes; the solver assumes the same


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def observe(name, polygon, doors, rng):
    """The room as the photo tier would report it: in its own frame (random quarter turn and shift)."""
    truth = Placement(float(rng.integers(4)) * np.pi / 2, tuple(rng.uniform(-20, 20, 2)))
    inverse = lambda xy: (np.asarray(xy, float) - truth.shift) @ rot(truth.angle)   # plan -> room frame
    local_doors = [Door(tuple(inverse(c)), tuple(rot(truth.angle).T @ np.asarray(n, float)), w) for c, n, w in doors]
    return Room(name, inverse(polygon), local_doors), truth


def l_shaped_home():
    """A (4 x 3) with B to its right and C above it, each through one door."""
    rooms = {
        "A": (box(0, 0, 4, 3), [((4.0, 1.5), (1, 0), 0.8), ((1.0, 3.0), (0, 1), 0.8)]),
        "B": (box(4 + T, 0, 7 + T, 3), [((4 + T, 1.5), (-1, 0), 0.8)]),
        "C": (box(0, 3 + T, 2, 6 + T), [((1.0, 3 + T), (0, -1), 0.8)]),
    }
    return rooms


def solve(rooms, seed=0):
    rng = np.random.default_rng(seed)
    names = list(rooms)
    observed = [observe(n, *rooms[n], rng) for n in names]
    result = stitch([o[0] for o in observed])
    return names, observed, result


def plan_error(names, observed, result, rooms):
    """Largest vertex distance between the stitched plan and the truth, in the first room's frame."""
    first = min(result.placements, key=lambda k: -abs(np.linalg.det(np.c_[observed[k][0].polygon[1] - observed[k][0].polygon[0],
                                                                         observed[k][0].polygon[2] - observed[k][0].polygon[1]])))
    anchor = next(k for k in result.placements if result.placements[k] == Placement(0.0, (0.0, 0.0)))
    to_anchor = lambda xy: (np.asarray(xy, float) - observed[anchor][1].shift) @ rot(observed[anchor][1].angle)
    worst = 0.0
    for k, name in enumerate(names):
        if k not in result.placements:
            continue
        got = result.placements[k].apply(observed[k][0].polygon)
        want = to_anchor(rooms[name][0])
        worst = max(worst, float(np.abs(got - want).max()))
    return worst


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_l_shaped_home_is_rejoined_exactly(seed):
    rooms = l_shaped_home()
    names, observed, result = solve(rooms, seed)
    assert result.islands == 1
    assert result.adjacency() == {frozenset((names.index("A"), names.index("B"))), frozenset((names.index("A"), names.index("C")))}
    assert result.overlap_m2 < 1e-6
    assert plan_error(names, observed, result, rooms) < 1e-6


def test_doors_to_elsewhere_and_a_false_door_do_not_break_it():
    rooms = l_shaped_home()
    rooms["B"][1].append(((7 + T, 1.5), (1, 0), 0.9))          # front door, leads outside
    rooms["C"][1].append(((2.0, 4.5), (1, 0), 0.8))            # a door the photos imagined
    names, observed, result = solve(rooms, seed=4)
    assert result.islands == 1
    assert result.adjacency() == {frozenset((names.index("A"), names.index("B"))), frozenset((names.index("A"), names.index("C")))}
    assert result.overlap_m2 < 1e-6


def test_a_door_seen_from_both_sides_confirms_a_second_pairing():
    # B also opens into C around the corner: once B and C are both placed, that door lines up by itself
    rooms = {
        "A": (box(0, 0, 4, 3), [((4.0, 1.5), (1, 0), 0.8), ((2.0, 3.0), (0, 1), 0.8)]),
        "B": (box(4 + T, 0, 7 + T, 3), [((4 + T, 1.5), (-1, 0), 0.8), ((5.5, 3.0), (0, 1), 0.7)]),
        "C": (box(0, 3 + T, 7 + T, 6), [((2.0, 3 + T), (0, -1), 0.8), ((5.5, 3 + T), (0, -1), 0.7)]),
    }
    names, observed, result = solve(rooms, seed=5)
    assert result.islands == 1 and len(result.pairs) == 3
    assert result.overlap_m2 < 1e-6
    assert plan_error(names, observed, result, rooms) < 1e-6


def test_rooms_whose_doors_cannot_match_stay_apart():
    rooms = {
        "A": (box(0, 0, 4, 3), [((4.0, 1.5), (1, 0), 1.6)]),     # a wide double door
        "B": (box(4 + T, 0, 7 + T, 3), [((4 + T, 1.5), (-1, 0), 0.7)]),
    }
    names, observed, result = solve(rooms, seed=6)
    assert result.islands == 2 and not result.pairs
    assert result.overlap_m2 < 1e-6


def test_a_room_outline_touching_itself_does_not_break_the_solver():
    # outline noise can make a room's outline touch itself; the solver keeps the largest part
    touching = np.array([[4 + T, 0], [6 + T, 0], [6 + T, 2], [8 + T, 2], [8 + T, 4], [6 + T, 4], [6 + T, 2], [4 + T, 2]], float)
    rooms = {
        "A": (box(0, 0, 4, 3), [((4.0, 1.0), (1, 0), 0.8)]),
        "B": (touching, [((4 + T, 1.0), (-1, 0), 0.8)]),
    }
    names, observed, result = solve(rooms, seed=7)
    assert result.islands == 1 and len(result.pairs) == 1

