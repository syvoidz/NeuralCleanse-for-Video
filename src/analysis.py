import numpy as np
from typing import List, Tuple

def outlier_detection(l1_norm_list: np.ndarray) -> Tuple[float, List[int]]:
    """
    使用中位数绝对偏差（MAD）算法来检测L1范数列表中的异常值。

    Args:
        l1_norm_list (np.ndarray): 每个类别反向工程出的触发器mask的L1范数列表。

    Returns:
        Tuple[float, List[int]]: 最小L1范数的异常指数，以及被标记为后门的类别索引列表。
    """
    consistency_constant = 1.4826  # for normal distribution
    median = np.median(l1_norm_list)
    # 健壮性：确保mad不为0
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    if mad < 1e-9:
        mad = 1e-9

    anomaly_scores = np.abs(l1_norm_list - median) / mad
    min_mad_score = float(anomaly_scores[np.argmin(l1_norm_list)])

    print("\n--- [PHASE 1] Anomaly Detection Results ---")
    print(f"Median L1 Norm: {median:.2f}, MAD: {mad:.2f}")
    print(f"Anomaly Index (of the minimum L1 norm): {min_mad_score:.2f}")

    # 异常的定义：分数 > 2 且 L1范数小于中位数
    flagged_indices = np.where((anomaly_scores > 2) & (l1_norm_list < median))[0]
    flagged_labels = [int(i) for i in flagged_indices]

    if flagged_labels:
        print(f"CONCLUSION: Backdoor DETECTED. Flagged Label(s): {flagged_labels}")
    else:
        print("CONCLUSION: No backdoor detected based on the anomaly threshold.")

    return min_mad_score, flagged_labels
