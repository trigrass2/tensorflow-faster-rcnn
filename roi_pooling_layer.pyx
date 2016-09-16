from __future__ import division
import numpy as np

cimport numpy as np

ctypedef np.float_t DTYPE_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

def roi_pooling_layer(np.ndarray[DTYPE_t, ndim=4] data, np.ndarray[np.int_t, ndim=2] rois):
    """
    Args:
        data: 1 x h x w x c
        rois: N x 5, [xmin, ymin, xmax, ymax], has been mapped on featmap
    """
    cdef int height = data.shape[1]
    cdef int width = data.shape[2]
    cdef int channel = data.shape[3]
    cdef int N = rois.shape[0]

    cdef int pooled_h = 7
    cdef int pooled_w = 7

    cdef np.ndarray[DTYPE_t, ndim=4] out = np.zeros([N, pooled_h, pooled_w, channel], dtype=np.float)
    cdef int i, ci, ph, pw, hi, wi
    cdef int roi_start_w, roi_start_h, roi_end_w, roi_end_h, roi_h, roi_w
    cdef int hstart, wstart, hend, wend
    cdef float bin_size_h, bin_size_w

    for i in range(N):
        roi_start_w = rois[i, 0]
        roi_start_h = rois[i, 1]
        roi_end_w = rois[i, 2]
        roi_end_h = rois[i, 3]

        roi_h = int_max(roi_end_h - roi_start_h + 1, 1)
        roi_w = int_max(roi_end_w - roi_start_w + 1, 1)

        bin_size_h = roi_h / pooled_h
        bin_size_w = roi_w / pooled_w

        for ci in range(channel):
            for ph in range(pooled_h):
                for pw in range(pooled_w):
                    hstart = int(np.floor(ph * bin_size_h))
                    wstart = int(np.floor(pw * bin_size_w))
                    hend = int(np.ceil((ph + 1) * bin_size_h))
                    wend = int(np.ceil((pw + 1) * bin_size_w))

                    hstart = int_min(int_max(hstart + roi_start_h, 0), height)
                    hend = int_min(int_max(hend + roi_start_h, 0), height)
                    wstart = int_min(int_max(wstart + roi_start_w, 0), width)
                    wend = int_min(int_max(wend + roi_start_w, 0), width)

                    for hi in range(hstart, hend):
                        for wi in range(wstart, wend):
                            if data[0, hi, wi, ci] > out[i, ph, pw, ci]:
                                out[i, ph, pw, ci] = data[0, hi, wi, ci]

    return out
