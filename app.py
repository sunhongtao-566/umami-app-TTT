
from flask import Flask, request, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

# Amino acid feature mappings
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
aa_properties = {
    'A': {'hydrophobicity': 1.8, 'polarity': 9.87, 'size': 89.1},  # Alanine
    'R': {'hydrophobicity': -4.5, 'polarity': 12.48, 'size': 174.2},  # Arginine
    'N': {'hydrophobicity': -3.5, 'polarity': 11.60, 'size': 132.1},  # Asparagine
    'D': {'hydrophobicity': -3.5, 'polarity': 3.90, 'size': 133.1},  # Aspartic acid
    'C': {'hydrophobicity': 2.5, 'polarity': 8.50, 'size': 121.2},  # Cysteine
    'E': {'hydrophobicity': -3.5, 'polarity': 4.25, 'size': 147.1},  # Glutamic acid
    'Q': {'hydrophobicity': -3.5, 'polarity': 10.50, 'size': 146.1},  # Glutamine
    'G': {'hydrophobicity': -0.4, 'polarity': 9.00, 'size': 75.1},  # Glycine
    'H': {'hydrophobicity': -3.2, 'polarity': 6.04, 'size': 155.2},  # Histidine
    'I': {'hydrophobicity': 4.5, 'polarity': 9.60, 'size': 131.2},  # Isoleucine
    'L': {'hydrophobicity': 3.8, 'polarity': 9.60, 'size': 131.2},  # Leucine
    'K': {'hydrophobicity': -3.9, 'polarity': 10.53, 'size': 146.2},  # Lysine
    'M': {'hydrophobicity': 1.9, 'polarity': 9.21, 'size': 149.2},  # Methionine
    'F': {'hydrophobicity': 2.8, 'polarity': 9.24, 'size': 165.2},  # Phenylalanine
    'P': {'hydrophobicity': -1.6, 'polarity': 10.64, 'size': 115.1},  # Proline
    'S': {'hydrophobicity': -0.8, 'polarity': 13.00, 'size': 105.1},  # Serine
    'T': {'hydrophobicity': -0.7, 'polarity': 13.00, 'size': 119.1},  # Threonine
    'W': {'hydrophobicity': -0.9, 'polarity': 9.41, 'size': 204.2},  # Tryptophan
    'Y': {'hydrophobicity': -1.3, 'polarity': 9.11, 'size': 181.2},  # Tyrosine
    'V': {'hydrophobicity': 4.2, 'polarity': 9.62, 'size': 117.1},  # Valine
}

def seq_to_features_with_properties(sequence, max_seq_length=200):
    k = len(amino_acids)
    pair_counts = np.zeros((k, k))
    hydro_sum, polarity_sum, size_sum = 0, 0, 0

    for i in range(len(sequence) - 1):
        if sequence[i] in aa_to_int and sequence[i + 1] in aa_to_int:
            pair_counts[aa_to_int[sequence[i]], aa_to_int[sequence[i + 1]]] += 1

    for aa in sequence:
        if aa in aa_properties:
            props = aa_properties[aa]
            hydro_sum += props['hydrophobicity']
            polarity_sum += props['polarity']
            size_sum += props['size']

    avg_hydro = hydro_sum / len(sequence)
    avg_polarity = polarity_sum / len(sequence)
    avg_size = size_sum / len(sequence)

    # Amino acid frequency
    aa_freq = np.zeros(len(amino_acids))
    for aa in sequence:
        if aa in aa_to_int:
            aa_freq[aa_to_int[aa]] += 1
    aa_freq = aa_freq / len(sequence)

    # Position weighted features
    weighted_freq = np.zeros(len(amino_acids))
    for i, aa in enumerate(sequence):
        if aa in aa_to_int:
            weight = 1 - (i / max_seq_length)  # Position weight: higher weight at the N-terminal
            weighted_freq[aa_to_int[aa]] += weight
    weighted_freq = weighted_freq / len(sequence)

    # Combine features to ensure 443 features are returned
    features = np.concatenate([
        pair_counts.flatten(),             # 400 dimensions
        [avg_hydro, avg_polarity, avg_size],  # 3 dimensions
        aa_freq,                           # 20 dimensions
        weighted_freq                      # 20 dimensions
    ])

    # Ensure the features are correct
    if len(features) != 443:
        raise ValueError(f"Expected 443 features, but got {len(features)} features.")

    return features

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        seq = request.form["sequence"]
        features = seq_to_features_with_properties(seq)  # Use the updated function
        pred = model.predict([features])[0]
        result = "Umami" if pred == 1 else "Not Umami"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
