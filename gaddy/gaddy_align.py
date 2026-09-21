"""Gaddy's monotonic DTW alignment, copied VERBATIM (by program, asserted byte-identical)
from silent_speech/align.py (github.com/dgaddy/silent_speech):
  time_warp: lines 5-14
  align_from_distances: lines 16-34
`matplotlib` is imported lazily inside the debug branch instead of at module level.
Do not edit the function bodies.

This is the alignment used by `dtw_loss` in Gaddy's transduction_model.py: for a silently
mouthed utterance the EMG has a different duration from the vocalised twin's audio (here
0.92x on average, 0.77-1.10x over the 5th-95th percentile), so the regression target can
only be paired frame-by-frame after a monotonic alignment, recomputed each step from the
current predictions' distance matrix.

KNOWN BOUNDARY QUIRK of this implementation, verified 2026-09-18 and deliberately NOT
changed (it is the code that produced the published results, and it costs one frame per
utterance): `time_warp` sets `dtw[0,1:] = inf` and `dtw[1:,0] = inf`, so the path may not
travel along the first row or column -- it must step diagonally off the origin.  A path
that should match several leading target frames to source frame 0 is therefore forbidden,
and the returned optimum can exceed the true minimum-cost monotonic path.  Measured on a
60-frame exact-copy test: one mismatched pair (target frame 1), accumulated cost 4.05 where
the true warp costs 0.  On a smooth non-degenerate warp the alignment recovers the true
time map with median error 0 and maximum error 1 frame.  At ~300 frames per utterance the
affected fraction is ~0.3 %."""
import numpy as np
from numba import jit


@jit(nopython=True)
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw


def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        import matplotlib.pyplot as plt  # noqa: F401  (lazy: debug branch only)
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results
