import pytest
import numpy as np
from src.health_monitoring.tracking.history import HistoryTracker


@pytest.fixture
def tracker():
    return HistoryTracker(window_size=3, update_period_frames=1, update_period_sec=1.0)


def test_initial_state(tracker):
    assert tracker.window_size == 3
    assert tracker.update_period_frames == 1
    assert tracker.update_period_sec == 1.0
    assert tracker.data == {}
    assert tracker.last_ids_list == []
    assert tracker.last_positions_list == []


def test_first_update_creates_keys_with_repeated_coords(tracker):
    ids = ['a', 'b']
    coords = [(1, 2), (3, 4)]
    tracker.update(ids, coords)

    assert set(tracker.data.keys()) == {'a', 'b'}
    for key, coord in zip(ids, coords):
        entry = tracker.data[key]
        assert entry['coords'] == [coord, coord, coord]
        assert entry['valid'] == [False, False, True]

    last_ids, last_positions = tracker.get_last_update_arrays()
    np.testing.assert_array_equal(last_ids, np.array(ids))
    np.testing.assert_array_equal(last_positions, np.array(coords))


def test_partial_update_and_non_updated_behavior(tracker):
    tracker.update(['a', 'b'], [(1, 1), (2, 2)])
    tracker.update(['a'], [(10, 10)])

    entry_a = tracker.data.get('a')
    assert entry_a is not None
    # coords -> [(1,1),(1,1),(10,10)]; valid -> [False, True, True]
    assert entry_a['coords'] == [(1, 1), (1, 1), (10, 10)]
    assert entry_a['valid'] == [False, True, True]

    entry_b = tracker.data.get('b')
    assert entry_b is not None
    # coords -> [(2,2),(2,2),(2,2)]; valid -> [False, True, False]
    assert entry_b['coords'] == [(2, 2), (2, 2), (2, 2)]
    assert entry_b['valid'] == [False, True, False]

    last_ids, last_positions = tracker.get_last_update_arrays()
    np.testing.assert_array_equal(last_ids, np.array(['a']))
    np.testing.assert_array_equal(last_positions, np.array([(10, 10)]))


def test_removal_after_all_invalid(tracker):
    tracker.update(['x'], [(0, 0)])
    assert 'x' in tracker.data
    assert tracker.data['x']['coords'] == [(0, 0), (0, 0), (0, 0)]
    assert tracker.data['x']['valid'] == [False, False, True]

    # Perform window_size consecutive empty updates to force removal of 'x' ...

    # 1
    tracker.update([], [])
    assert 'x' in tracker.data
    assert tracker.data['x']['coords'] == [(0, 0), (0, 0), (0, 0)]
    assert tracker.data['x']['valid'] == [False, True, False]
    # 2
    tracker.update([], [])
    assert 'x' in tracker.data
    assert tracker.data['x']['coords'] == [(0, 0), (0, 0), (0, 0)]
    assert tracker.data['x']['valid'] == [True, False, False]

    # 3
    tracker.update([], [])
    assert 'x' not in tracker.data


def test_get_ids_history_and_shapes(tracker):
    tracker.update(['a', 'b'], [(1, 2), (3, 4)])
    tracker.update(['c'], [(5, 6)])
    tracker.update(['d'], [(7, 8)])
    hist = tracker.get_ids_history(['c'])
    coords_arr = hist['c']['coords']
    valid_arr = hist['c']['valid']
    assert coords_arr.shape == (2, tracker.window_size)
    assert valid_arr.shape == (1, tracker.window_size)
    expected_coords = np.array([[5, 5, 5], [6, 6, 6]])
    expected_valid = np.array([[False, True, False]])
    np.testing.assert_array_equal(coords_arr, expected_coords)
    np.testing.assert_array_equal(valid_arr, expected_valid)


def test_get_ids_history_invalid_id_raises(tracker):
    # 'nonexistent' not in history
    with pytest.raises(ValueError):
        tracker.get_ids_history(['nonexistent'])

    # 'b' not in history
    tracker.update(['a'], [(1, 1)])
    with pytest.raises(ValueError):
        tracker.get_ids_history(['a', 'b'])

    # empty list of ids not acceptable
    with pytest.raises(ValueError):
        tracker.get_ids_history([])


def test_get_and_aggregate_ids_history(tracker):
    tracker.update(['a', 'b'], [(1, 2), (3, 4)])
    tracker.update(['c', 'b'], [(5, 6), (7, 8)])
    coords_array, valid_array = tracker.get_and_aggregate_ids_history(['a', 'b'])
    assert coords_array.shape == (2, 2, tracker.window_size)
    assert valid_array.shape == (2, 1, tracker.window_size)

    np.testing.assert_array_equal(coords_array[0], np.array([[1, 1, 1], [2, 2, 2]]))
    np.testing.assert_array_equal(valid_array[0], np.array([[False, True, False]]))
    np.testing.assert_array_equal(coords_array[1], np.array([[3, 3, 7], [4, 4, 8]]))
    np.testing.assert_array_equal(valid_array[1], np.array([[False, True, True]]))


def test_get_and_aggregate_ids_history_single(tracker):
    tracker.update(['a', 'b'], [(1, 2), (3, 4)])
    tracker.update(['c', 'b'], [(5, 6), (7, 8)])
    coords_array, valid_array = tracker.get_and_aggregate_ids_history(['c'])
    assert coords_array.shape == (1, 2, tracker.window_size)
    assert valid_array.shape == (1, 1, tracker.window_size)

    np.testing.assert_array_equal(coords_array[0], np.array([[5, 5, 5], [6, 6, 6]]))
    np.testing.assert_array_equal(valid_array[0], np.array([[False, False, True]]))


def test_get_last_update_arrays_empty_and_after_updates(tracker):
    last_ids, last_positions = tracker.get_last_update_arrays()
    np.testing.assert_array_equal(last_ids, np.array([]))
    np.testing.assert_array_equal(last_positions, np.array([]))

    tracker.update(['A', 'B'], [(9, 9), (8, 8)])
    last_ids, last_positions = tracker.get_last_update_arrays()
    np.testing.assert_array_equal(last_ids, np.array(['A', 'B']))
    np.testing.assert_array_equal(last_positions, np.array([(9, 9), (8, 8)]))
