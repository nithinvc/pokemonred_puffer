"""A float32 array whose full `np.sum` is maintained incrementally with bit-identical results.

`np.sum` over the 484x476 reward exploration map ran every env step (~25 us), although at most
one cell changes per step. NumPy reduces a large contiguous float32 array as a sequential sum of
pairwise-summed blocks of `_CHUNK` elements, so we cache one sum per block, recompute only blocks
whose cells were written, and fold the block sums in the same order. The result is the same
float32 value np.sum returns.

Writes are tracked in `__setitem__` (one block for a single-cell write, everything otherwise) and
for any ufunc that writes into the array in place (e.g. `arr *= 0` in the exploration wrappers).
If the block model does not reproduce np.sum on this NumPy build, tracking is disabled and
np.sum runs as before.
"""

import numpy as np

_CHUNK = 8192


def _chunked_sum(flat: np.ndarray) -> np.float32:
    sums = np.array(
        [np.add.reduce(flat[i : i + _CHUNK]) for i in range(0, flat.size, _CHUNK)],
        dtype=np.float32,
    )
    return np.add.accumulate(sums)[-1]


def _model_holds() -> bool:
    rng = np.random.default_rng(1234)
    for shape in [(484, 476), (300, 1000), (70000,)]:
        a = np.zeros(shape, np.float32)
        idx = rng.integers(0, a.size, 4000)
        a.flat[idx] = rng.choice(
            np.array([0.5, 1.0, 9.539340019226074, 0.15, 3.3], np.float32), 4000
        )
        if np.sum(a) != _chunked_sum(a.reshape(-1)):
            return False
        a = rng.random(shape).astype(np.float32) * 7
        if np.sum(a) != _chunked_sum(a.reshape(-1)):
            return False
    return True


TRACKING_ENABLED = _model_holds()


class SumTrackedArray(np.ndarray):
    """C-contiguous float32 ndarray; `np.sum(arr)` is answered from per-block caches."""

    def __new__(cls, shape):
        obj = np.zeros(shape, dtype=np.float32).view(cls)
        obj._track = TRACKING_ENABLED
        n_chunks = (obj.size + _CHUNK - 1) // _CHUNK
        obj._sums = np.zeros(n_chunks, dtype=np.float32)
        obj._dirty = set()
        obj._dirty_all = True
        obj._total = np.float32(0)
        return obj

    def __array_finalize__(self, obj):
        # Views, slices and copies are plain (untracked) arrays of this class.
        self._track = False

    def __reduce__(self):
        # pickle as a plain ndarray
        return np.asarray(self).__reduce__()

    def _base(self) -> np.ndarray:
        return self.view(np.ndarray)

    def _mark(self, key) -> None:
        if not self._track or self._dirty_all:
            return
        if (
            type(key) is tuple
            and len(key) == self.ndim
            and all(type(k) is int or isinstance(k, np.integer) for k in key)
        ):
            flat = 0
            for k, n in zip(key, self.shape):
                k = int(k)
                if k < 0:
                    k += n
                if not 0 <= k < n:
                    return  # IndexError raised by numpy below; nothing written
                flat = flat * n + k
            self._dirty.add(flat // _CHUNK)
        else:
            self._dirty_all = True

    def __setitem__(self, key, value):
        self._mark(key)
        self._base()[key] = value

    def _exact_sum(self) -> np.float32:
        flat = self._base().reshape(-1)
        if self._dirty_all:
            for c in range(self._sums.size):
                self._sums[c] = np.add.reduce(flat[c * _CHUNK : (c + 1) * _CHUNK])
        else:
            for c in self._dirty:
                self._sums[c] = np.add.reduce(flat[c * _CHUNK : (c + 1) * _CHUNK])
        if self._dirty_all or self._dirty:
            self._total = np.add.accumulate(self._sums)[-1]
        self._dirty.clear()
        self._dirty_all = False
        return self._total

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if (
            self._track
            and ufunc is np.add
            and method == "reduce"
            and len(inputs) == 1
            and inputs[0] is self
            and out is None
            and kwargs.get("axis", 0) is None
            and kwargs.get("dtype") is None
            and not kwargs.get("keepdims", False)
            and "initial" not in kwargs
            and kwargs.get("where", True) is True
        ):
            return self._exact_sum()
        args = [x.view(np.ndarray) if isinstance(x, SumTrackedArray) else x for x in inputs]
        if out is not None:
            for o in out:
                if isinstance(o, SumTrackedArray) and o._track:
                    o._dirty_all = True
            kwargs["out"] = tuple(
                o.view(np.ndarray) if isinstance(o, SumTrackedArray) else o for o in out
            )
        result = getattr(ufunc, method)(*args, **kwargs)
        if out is not None and len(out) == 1 and isinstance(out[0], SumTrackedArray):
            return out[0]
        return result
