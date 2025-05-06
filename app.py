from flask import Flask, request, render_template
import numpy as np
import joblib

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型
model = joblib.load("model.pkl")


# 定义氨基酸到整数的映射字典
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}

# 为每个氨基酸分配化学性质
aa_properties = {
    'A': {'hydrophobicity': 1.8,  'polarity': 9.87,  'size': 89.1},    # Alanine
    'R': {'hydrophobicity': -4.5, 'polarity': 12.48, 'size': 174.2},   # Arginine
    'N': {'hydrophobicity': -3.5, 'polarity': 11.60, 'size': 132.1},   # Asparagine
    'D': {'hydrophobicity': -3.5, 'polarity': 3.90,  'size': 133.1},   # Aspartic acid
    'C': {'hydrophobicity': 2.5,  'polarity': 8.50,  'size': 121.2},   # Cysteine
    'E': {'hydrophobicity': -3.5, 'polarity': 4.25,  'size': 147.1},   # Glutamic acid
    'Q': {'hydrophobicity': -3.5, 'polarity': 10.50, 'size': 146.1},   # Glutamine
    'G': {'hydrophobicity': -0.4, 'polarity': 9.00,  'size': 75.1},    # Glycine
    'H': {'hydrophobicity': -3.2, 'polarity': 6.04,  'size': 155.2},   # Histidine
    'I': {'hydrophobicity': 4.5,  'polarity': 9.60,  'size': 131.2},   # Isoleucine
    'L': {'hydrophobicity': 3.8,  'polarity': 9.60,  'size': 131.2},   # Leucine
    'K': {'hydrophobicity': -3.9, 'polarity': 10.53, 'size': 146.2},   # Lysine
    'M': {'hydrophobicity': 1.9,  'polarity': 9.21,  'size': 149.2},   # Methionine
    'F': {'hydrophobicity': 2.8,  'polarity': 9.24,  'size': 165.2},   # Phenylalanine
    'P': {'hydrophobicity': -1.6, 'polarity': 10.64, 'size': 115.1},   # Proline
    'S': {'hydrophobicity': -0.8, 'polarity': 13.00, 'size': 105.1},   # Serine
    'T': {'hydrophobicity': -0.7, 'polarity': 13.00, 'size': 119.1},   # Threonine
    'W': {'hydrophobicity': -0.9, 'polarity': 9.41,  'size': 204.2},   # Tryptophan
    'Y': {'hydrophobicity': -1.3, 'polarity': 9.11,  'size': 181.2},   # Tyrosine
    'V': {'hydrophobicity': 4.2,  'polarity': 9.62,  'size': 117.1},   # Valine
}

def seq_to_features_with_properties_optimized(sequence, max_seq_length=200):
    k = len(amino_acids)
    pair_counts = np.zeros((k, k))
    hydro_values, polarity_values, size_values = [], [], []

    for i in range(len(sequence) - 1):
        if sequence[i] in aa_to_int and sequence[i + 1] in aa_to_int:
            pair_counts[aa_to_int[sequence[i]], aa_to_int[sequence[i + 1]]] += 1

    for aa in sequence:
        if aa in aa_properties:
            props = aa_properties[aa]
            hydro_values.append(props['hydrophobicity'])
            polarity_values.append(props['polarity'])
            size_values.append(props['size'])

    avg_hydro = np.mean(hydro_values)
    avg_polarity = np.mean(polarity_values)
    avg_size = np.mean(size_values)
    std_hydro = np.std(hydro_values)
    std_polarity = np.std(polarity_values)
    std_size = np.std(size_values)
    range_hydro = np.max(hydro_values) - np.min(hydro_values)
    range_polarity = np.max(polarity_values) - np.min(polarity_values)
    range_size = np.max(size_values) - np.min(size_values)
    # 添加：结构感知的局部滑窗统计特征（窗口大小为3）
    window_size = 3
    half = window_size // 2
    local_hydro, local_polarity, local_size = [], [], []

    for i in range(len(sequence)):
        win_h, win_p, win_s = [], [], []
        for j in range(i - half, i + half + 1):
            if 0 <= j < len(sequence) and sequence[j] in aa_properties:
                props = aa_properties[sequence[j]]
                win_h.append(props['hydrophobicity'])
                win_p.append(props['polarity'])
                win_s.append(props['size'])
        # 计算窗口内的平均值（如果为空则为0）
        local_hydro.append(np.mean(win_h) if win_h else 0)
        local_polarity.append(np.mean(win_p) if win_p else 0)
        local_size.append(np.mean(win_s) if win_s else 0)

    # 提取滑窗特征的全局统计量（也可以改为 max, min, std 等）
    avg_local_hydro = np.mean(local_hydro)
    std_local_hydro = np.std(local_hydro)
    avg_local_polarity = np.mean(local_polarity)
    std_local_polarity = np.std(local_polarity)
    avg_local_size = np.mean(local_size)
    std_local_size = np.std(local_size)

    aa_freq = np.zeros(k)
    for aa in sequence:
        if aa in aa_to_int:
            aa_freq[aa_to_int[aa]] += 1
    aa_freq = aa_freq / len(sequence)

    weighted_freq = np.zeros(k)
    for i, aa in enumerate(sequence):
        if aa in aa_to_int:
            weight = 1 - (i / max_seq_length)
            weighted_freq[aa_to_int[aa]] += weight
    weighted_freq = weighted_freq / len(sequence)

    freq_std = np.std(aa_freq)

    # ✅ 保证100个常见三肽
    common_trimers = [a + b + c for a in 'ACDEFGHIKLMNPQRSTVWY' for b in 'ACDEFGHIKLMNPQRSTVWY' for c in 'ACDEFGHIKLMNPQRSTVWY']
    common_trimers = common_trimers[:100]  # 截取前100个即可
    trimer_counts = dict.fromkeys(common_trimers, 0)
    for i in range(len(sequence) - 2):
        trimer = sequence[i:i+3]
        if trimer in trimer_counts:
            trimer_counts[trimer] += 1
    trimer_features = np.array([trimer_counts[tri] for tri in common_trimers]) / max(1, len(sequence) - 2)

    len_norm = len(sequence) / max_seq_length
    # 构建疏水性数值序列
    hydro_seq = [aa_properties[aa]['hydrophobicity'] for aa in sequence if aa in aa_properties]

    # Padding 处理，保持最小长度
    min_len = 20
    if len(hydro_seq) < min_len:
        hydro_seq += [0.0] * (min_len - len(hydro_seq))

    # 简化版 MLBP：3点滑窗中心与两边比较，构建二进制编码
    def simple_mlbp(seq):
        patterns = []
        for i in range(1, len(seq) - 1):
            center = seq[i]
            code = 0
            code |= (seq[i - 1] >= center) << 1
            code |= (seq[i + 1] >= center)
            patterns.append(code)
        # 统计编码 00, 01, 10, 11 的出现次数
        counts = [patterns.count(i) for i in range(4)]
        return np.array(counts) / max(1, len(patterns))  # 归一化

    mlbp_features = simple_mlbp(hydro_seq)

    features = np.concatenate([
        pair_counts.flatten(),  # 400
        [avg_hydro, avg_polarity, avg_size],  # 3
        [std_hydro, std_polarity, std_size],  # 3
        [range_hydro, range_polarity, range_size],  # 3
        aa_freq,  # 20
        weighted_freq,  # 20
        [freq_std],  # 1
        trimer_features,  # 100
        [len_norm],  # 1
        [avg_local_hydro, std_local_hydro,  # ✅ 新增结构感知特征
         avg_local_polarity, std_local_polarity,
         avg_local_size, std_local_size]
    ])

    if len(features) != 557:
        raise ValueError(f"Expected 557 features, but got {len(features)} features.")

    return features

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form['sequence'].strip().upper()
    try:
        features = seq_to_features_with_properties_optimized(sequence)
        prediction = model.predict([features])[0]
        label = 'Umami' if prediction == 1 else 'Not Umami'
    except Exception as e:
        label = f"Error: {str(e)}"
    return render_template('index.html', result=label)

if __name__ == '__main__':
    app.run(debug=True)

