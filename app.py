import streamlit as st
import numpy as np
import pandas as pd
import joblib
from math import acos, degrees

# =========================
# MODEL PATHS (SYMMETRY BASED)
# =========================
MODEL_PATHS = {
    "D4h": {
        "Ucal": "D4h_U_GB_model.joblib",
        "B20": "D4h_B20_GB_model.joblib",
    },
    "D5h": {
        "Ucal": "D5h_U_GB_model.joblib",
        "B20": "D5h_B20_GB_model.joblib",
    },
    "D6h": {
        "Ucal": "D6h_U_GB_model.joblib",
        "B20": "D6h_B20_GB_model.joblib",
    }
}

# Global models (NO gz)
UEFF_MODEL = "Ueff_GB_model.joblib"
TAU_MODEL = "tio_no_gz_model.joblib"

st.set_page_config(page_title="Dy Magnetic Predictor", layout="wide")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models(sym):
    paths = MODEL_PATHS[sym]
    return (
        joblib.load(paths["Ucal"]),
        joblib.load(paths["B20"]),
        joblib.load(UEFF_MODEL),
        joblib.load(TAU_MODEL),
    )

# =========================
# FILE READER
# =========================
def read_xyz(file):
    lines = file.read().decode().splitlines()
    atoms, coords = [], []

    for line in lines:
        p = line.split()
        if len(p) < 4:
            continue
        try:
            xyz = list(map(float, p[1:4]))
        except:
            continue

        atoms.append(p[0])
        coords.append(xyz)

    return atoms, np.array(coords)

def find_dy(atoms):
    for i, a in enumerate(atoms):
        if str(a).lower() == "dy" or str(a) == "66":
            return i
    return None

# =========================
# GEOMETRY
# =========================
def dist(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    ba = a - b
    bc = c - b
    return degrees(
        acos(
            np.clip(
                np.dot(ba, bc) /
                (np.linalg.norm(ba) * np.linalg.norm(bc)),
                -1, 1
            )
        )
    )

# =========================
# EQUATORIAL SELECTION
# =========================
def get_equatorial_atoms(atoms, coords, dy_idx, ax1, ax2):

    Dy = coords[dy_idx]
    Ax1 = coords[ax1]
    Ax2 = coords[ax2]

    candidates = []

    for i, coord in enumerate(coords):
        if i in [dy_idx, ax1, ax2]:
            continue

        atom = atoms[i]
        d = dist(Dy, coord)

        if atom.upper() == "H":
            if not (1.8 <= d <= 2.0):
                continue
        else:
            if not (1.9 <= d <= 3.5):
                continue

        ang1 = angle(Ax1, Dy, coord)
        ang2 = angle(Ax2, Dy, coord)

        if 60 <= ang1 <= 150 and 60 <= ang2 <= 150:
            candidates.append((i + 1, atom, d))

    return sorted(candidates, key=lambda x: x[2])

# =========================
# UI
# =========================
st.title("Dy Magnetic Property Predictor (No gz)")

xyz_file = st.file_uploader("Upload XYZ file", type=["xyz"])

col1, col2 = st.columns(2)
ax1_input = col1.number_input("Axial atom index 1", min_value=1)
ax2_input = col2.number_input("Axial atom index 2", min_value=1)

symmetry = st.selectbox("Select symmetry", ["D4h", "D5h", "D6h"])

# =========================
# RUN
# =========================
if st.button("Predict"):

    atoms, coords = read_xyz(xyz_file)
    dy_idx = find_dy(atoms)

    ax1 = int(ax1_input) - 1
    ax2 = int(ax2_input) - 1

    Dy = coords[dy_idx]
    Ax1 = coords[ax1]
    Ax2 = coords[ax2]

    # Axial
    A1 = dist(Dy, Ax1)
    A2 = dist(Dy, Ax2)

    if A1 > A2:
        A1, A2 = A2, A1

    BA = abs(180 - angle(Ax1, Dy, Ax2))

    # Equatorial
    candidates = get_equatorial_atoms(atoms, coords, dy_idx, ax1, ax2)

    CN = {"D4h": 4, "D5h": 5, "D6h": 6}[symmetry]
    selected = candidates[:CN]
    eq_distances = sorted([x[2] for x in selected])

    # E handling
    if symmetry == "D4h":
        E = eq_distances[:4]
    elif symmetry == "D5h":
        E = [eq_distances[0], eq_distances[1], eq_distances[2],
             np.mean(eq_distances[3:5])]
    elif symmetry == "D6h":
        E = [eq_distances[0], eq_distances[1], eq_distances[2],
             np.mean(eq_distances[3:6])]

    # Load models
    u_model, b20_model, ueff_model, tau_model = load_models(symmetry)

    # Base features
    base_features = np.array([
        A1, A2, BA,
        *eq_distances
    ]).reshape(1, -1)

    # Predict intermediate
    Ucal = u_model.predict(base_features)[0]
    B20 = b20_model.predict(base_features)[0]

    # Final features (NO gz)
    final_features = np.array([
        CN, A1, A2, BA,
        E[0], E[1], E[2], E[3],
        B20, Ucal
    ]).reshape(1, -1)

    # Final predictions
    Ueff = ueff_model.predict(final_features)[0]
    log_tau = tau_model.predict(final_features)[0]

    # Output
    st.subheader("Predictions")

    st.success(f"Ucal: {Ucal:.2f}")
    st.success(f"B20: {B20:.4f}")
    st.success(f"Ueff: {Ueff:.2f}")
    st.success(f"log(tau0): {log_tau:.4f}")
