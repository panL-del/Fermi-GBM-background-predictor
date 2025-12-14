# 将cspec数据重新分bin到固定时间间隔（例如5s）
import numpy as np
import pandas as pd

#每一行的光子计数，应该是在一小段间隔时间内的光子计数，如果已有的数据中有start_time和end_time则不许要下面的函数
def estimate_bin_edges_from_centers(met):
    """
    从时间中心 met 估算每个 bin 的起止时间。
    假设 bin 边界在相邻中心的中点。
    """
    met = np.asarray(met)
    n = len(met)
    edges = np.empty(n + 1)

    # 内部边界：相邻中心的中点
    edges[1:-1] = (met[:-1] + met[1:]) / 2.0

    # 首尾边界：用首/末间隔外推
    if n > 1:
        edges[0] = met[0] - (edges[1] - met[0])
        edges[-1] = met[-1] + (met[-1] - edges[-2])
    else:
        # 只有一个点，无法估计宽度 → 设为默认值（如 4.096）
        edges[0] = met[0] - 2.048
        edges[1] = met[0] + 2.048

    start = edges[:-1]
    stop  = edges[1:]
    return start, stop



def rebin_cspec_to_5s(cspec_time, cspec_timedel, cspec_counts, t0=None, t1=None, dt_new=5.0):
    """
    将 cspec 的 counts 重分 bin 到固定 5s 网格（按时间重叠比例分配）。
    
    Parameters:
        cspec_time      : array, 原始 cspec TIME（中心时间）
        cspec_timedel   : array, 原始 cspec TIMEDEL（bin 宽度）
        cspec_counts    : array of shape (N,) or (N, E), 光子计数（可单能道或多能道）
        t0, t1          : 新时间网格的起止时间（MET）。若为 None，则自动取 cspec 覆盖范围
        dt_new          : 新 bin 宽度，默认 5.0 秒
    
    Returns:
        new_time_centers : 新 bin 的中心时间
        new_counts       : 重分 bin 后的 counts（形状匹配输入）
    """
    cspec_time = np.asarray(cspec_time)
    cspec_timedel = np.asarray(cspec_timedel)
    cspec_counts = np.asarray(cspec_counts)

    # 原始 bin 边界
    orig_start = cspec_time - cspec_timedel / 2.0
    orig_stop  = cspec_time + cspec_timedel / 2.0

    # 确定新时间范围
    if t0 is None:
        t0 = np.floor(orig_start.min() / dt_new) * dt_new
    if t1 is None:
        t1 = np.ceil(orig_stop.max() / dt_new) * dt_new

    # 新的 5s bin 边界
    new_edges = np.arange(t0, t1 + dt_new, dt_new)  # [t0, t0+5, t0+10, ..., t1]
    n_new = len(new_edges) - 1
    new_time_centers = (new_edges[:-1] + new_edges[1:]) / 2.0

    # 初始化输出
    if cspec_counts.ndim == 1:
        new_counts = np.zeros(n_new)
    else:
        new_counts = np.zeros((n_new, cspec_counts.shape[1]))

    # 遍历每个原始 bin，将其 counts 按重叠比例分配到新 bin
    for i in range(len(cspec_time)):
        t_orig_start = orig_start[i]
        t_orig_stop  = orig_stop[i]
        counts_i = cspec_counts[i]

        # 找出所有与当前原始 bin 重叠的新 bin
        overlaps = (new_edges[:-1] < t_orig_stop) & (new_edges[1:] > t_orig_start)

        if not np.any(overlaps):
            continue

        # 对每个重叠的新 bin，计算重叠时长
        for j in np.where(overlaps)[0]:
            overlap_start = max(t_orig_start, new_edges[j])
            overlap_stop  = min(t_orig_stop,  new_edges[j+1])
            overlap_dur   = overlap_stop - overlap_start
            total_orig_dur = t_orig_stop - t_orig_start

            weight = overlap_dur / total_orig_dur  # 分配比例
            new_counts[j] += weight * counts_i

    return new_time_centers, new_counts







