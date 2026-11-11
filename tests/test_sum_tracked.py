"""SumTrackedArray must return exactly np.sum's float32 value after any sequence of the writes the
env and the exploration wrappers make (no ROM needed)."""

import pickle

import numpy as np
import pytest

from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE
from pokemonred_puffer.sum_tracked import TRACKING_ENABLED, SumTrackedArray


def _plain_sum(a):
    return np.add.reduce(np.asarray(a).reshape(-1))


def test_block_model_holds_on_this_numpy():
    # If this fails, tracking silently falls back to np.sum: still correct, but slow.
    assert TRACKING_ENABLED


@pytest.mark.parametrize("shape", [GLOBAL_MAP_SHAPE, (300, 1000), (70000,)])
def test_random_operation_sequences(shape):
    rng = np.random.default_rng(0)
    a = SumTrackedArray(shape)
    ref = np.zeros(shape, dtype=np.float32)
    values = np.array([0.5, 1.0, 0.15, 3.3, 9.539340019226074, 1e-3], dtype=np.float32)
    for step in range(5000):
        op = rng.random()
        if op < 0.80:  # single-cell write, as in update_seen_coords
            idx = tuple(int(rng.integers(0, n)) for n in shape)
            v = values[rng.integers(0, len(values))]
            a[idx] = v
            ref[idx] = v
        elif op < 0.84:  # negative / numpy-integer indices
            idx = tuple(np.int64(-int(rng.integers(1, n + 1))) for n in shape)
            a[idx] = 2.0
            ref[idx] = 2.0
        elif op < 0.88:  # in-place scaling (exploration wrappers' forgetting factor)
            a *= np.float32(0.995)
            ref *= np.float32(0.995)
        elif op < 0.92:  # masked assignment from a clipped copy
            m = ref > 0
            a[m] = np.clip(a[m], 0.15, 1)
            ref[m] = np.clip(ref[m], 0.15, 1)
        elif op < 0.94:  # zeroing on reset
            a *= 0
            ref *= 0
        elif op < 0.96:  # slice assignment
            sl = slice(
                int(rng.integers(0, shape[0] // 2)), int(rng.integers(shape[0] // 2, shape[0]))
            )
            a[sl] = 0.25
            ref[sl] = 0.25
        elif op < 0.98:  # ufunc with explicit out=
            np.multiply(a, np.float32(1.01), out=a)
            np.multiply(ref, np.float32(1.01), out=ref)
        else:  # reads must not disturb tracking
            _ = a[0] + 1
            _ = a.max()
        if step % 7 == 0 or op >= 0.84:
            got = np.sum(a)
            assert got == np.sum(ref), step
            assert got.dtype == np.float32
            assert got == _plain_sum(a)


def test_out_of_range_write_raises_and_keeps_sum():
    a = SumTrackedArray((10, 10))
    a[1, 1] = 1.0
    with pytest.raises(IndexError):
        a[10, 0] = 5.0
    assert np.sum(a) == np.float32(1.0)


def test_other_reductions_and_views_are_plain():
    a = SumTrackedArray((50, 60))
    a[3, 4] = 2.0
    a[10, 11] = 0.5
    assert np.sum(a, axis=0).sum() == np.float32(2.5)
    assert np.sum(a, dtype=np.float64) == 2.5
    view = a[:10]
    assert np.sum(view) == np.float32(2.0)
    assert np.sum(a) == np.float32(2.5)


def test_pickles_as_plain_ndarray():
    a = SumTrackedArray((20, 20))
    a[1, 2] = 3.0
    b = pickle.loads(pickle.dumps(a))
    assert type(b) is np.ndarray
    assert np.array_equal(b, np.asarray(a))
