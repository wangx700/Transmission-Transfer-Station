"""缩小版QKV NaN probe；仅用标准库，不执行真实Bridge或集合通信。"""

import math


def split_qkv(rows, groups, q_per_group, head_dim):
    """按Megatron分组排列拆出HF Q/K/V行，保留NaN占位。"""
    q, k, v = [], [], []
    group_rows = (q_per_group + 2) * head_dim
    q_rows = q_per_group * head_dim
    for group in range(groups):
        start = group * group_rows
        q.extend(rows[start : start + q_rows])
        k.extend(rows[start + q_rows : start + q_rows + head_dim])
        v.extend(rows[start + q_rows + head_dim : start + group_rows])
    return q, k, v


def main():
    # 教学维度：H=4，Q heads=4，KV heads=2，D=2，TP=2。
    # 每个rank一个query group：[Q0,Q1,K,V]，共8行。
    width, local_rows = 4, 8
    local = [[math.nan] * width for _ in range(local_rows)]
    local[0][1] = 0.5  # TP1本地Q
    local[4][2] = 0.7  # TP1本地K
    other = [[math.nan] * width for _ in range(local_rows)]
    synthetic_gather = other + local  # TP1必须放在真实rank位置
    outs = split_qkv(synthetic_gather, groups=2, q_per_group=2, head_dim=2)
    indices, values, counts = [], [], []
    for name, rows in zip(('q_proj', 'k_proj', 'v_proj'), outs):
        flat = [value for row in rows for value in row]
        selected = [(i, value) for i, value in enumerate(flat) if not math.isnan(value)]
        print(f'{name}: shape=({len(rows)},{width}), entries={selected}')
        counts.append(len(selected))
        indices.extend(i for i, _ in selected)
        values.extend(value for _, value in selected)
    assert counts == [1, 1, 0]
    assert indices == [17, 10]
    assert values == [0.5, 0.7]
    # 模拟接收端NaN底板和masked copy：未变化位置保留旧值1.0。
    q_patch = [math.nan] * 32
    q_patch[17] = 0.5
    old_q = [1.0] * 32
    new_q = [old if math.isnan(src) else src for old, src in zip(old_q, q_patch)]
    assert new_q[17] == 0.5
    assert sum(a != b for a, b in zip(old_q, new_q)) == 1
    print(f'counts={counts}, hf_idx={indices}, hf_val={values}')
    print('PASS: probe坐标、slot分段、masked copy保留未变化值')


if __name__ == '__main__':
    main()
