import streamlit as st
import numpy as np
import pandas as pd
import joblib
from math import acos, degrees

# =========================
# MODEL PATHS (UPDATED)
# =========================
U_MODEL = "Ucal_GB_model.joblib"
B20_MODEL = "B20_GB_model.joblib"
TAU_MODEL = "tio_no_gz_model.joblib"   # 🔴 NEW MODEL (10 features)

st.set_page_config(page_title="Dy Magnetic Predictor", layout="wide")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    u_model = joblib.load(U_MODEL)
    b20_model = joblib.load(B20_MODEL)
    tau_model = joblib.load(TAU_MODEL)
    return u_model, b20_model, tau_model

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
# GEOMETRY FUNCTIONS
# =========================
def dist(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):
    ba = a - b
    bc = c - b
    return degrees(
        acos(
            np.clip(
                np.dot(ba, bc)
                / (np.linalg.norm(ba) * np.linalg.norm(bc)),
                -1,
                1,
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

    candidates = sorted(candidates, key=lambda x: x[2])
    return candidates

# =========================
# UI
# =========================
st.title("Dy Magnetic Property Predictor (No gz)")

xyz_file = st.file_uploader("Upload XYZ file", type=["xyz"])

col1, col2 = st.columns(2)
ax1_input = col1.number_input("Axial atom index 1 (1-based)", min_value=1)
ax2_input = col2.number_input("Axial atom index 2 (1-based)", min_value=1)

symmetry = st.selectbox("Select symmetry", ["D4h", "D5h", "D6h"])

# =========================
# RUN
# =========================
if st.button("Predict"):

    if xyz_file is None:
        st.error("Upload XYZ file")
        st.stop()

    atoms, coords = read_xyz(xyz_file)
    dy_idx = find_dy(atoms)

    if dy_idx is None:
        st.error("Dy not found")
        st.stop()

    ax1 = int(ax1_input) - 1
    ax2 = int(ax2_input) - 1

    Dy = coords[dy_idx]
    Ax1 = coords[ax1]
    Ax2 = coords[ax2]

    # =========================
    # AXIAL PARAMETERS
    # =========================
    A1 = dist(Dy, Ax1)
    A2 = dist(Dy, Ax2)

    # Ensure A1 < A2
    if A1 > A2:
        A1, A2 = A2, A1

    BA = abs(180 - angle(Ax1, Dy, Ax2))

    # =========================
    # EQUATORIAL
    # =========================
    candidates = get_equatorial_atoms(atoms, coords, dy_idx, ax1, ax2)

    CN = {"D4h": 4, "D5h": 5, "D6h": 6}[symmetry]

    if len(candidates) < CN:
        st.error(f"Only {len(candidates)} equatorial atoms found (need {CN})")
        st.stop()

    selected = candidates[:CN]
    eq_distances = sorted([x[2] for x in selected])

    # =========================
    # DISPLAY STRUCTURE
    # =========================
    st.subheader("Structural Parameters")

    st.write("A1:", A1)
    st.write("A2:", A2)
    st.write("BA:", BA)

    for i, val in enumerate(eq_distances):
        st.write(f"E{i+1}:", val)

    # =========================
    # E HANDLING (IMPORTANT)
    # =========================
    if symmetry == "D4h":
        E = eq_distances[:4]

    elif symmetry == "D5h":
        E = [
            eq_distances[0],
            eq_distances[1],
            eq_distances[2],
            np.mean(eq_distances[3:5])
        ]

    elif symmetry == "D6h":
        E = [
            eq_distances[0],
            eq_distances[1],
            eq_distances[2],
            np.mean(eq_distances[3:6])
        ]

    # =========================
    # LOAD MODELS
    # =========================
    u_model, b20_model, tau_model = load_models()

    # =========================
    # BASE FEATURES
    # =========================
    base_features = np.array([
        CN, A1, A2, BA,
        E[0], E[1], E[2], E[3]
    ]).reshape(1, -1)

    # =========================
    # INTERMEDIATE PREDICTION
    # =========================
    Ucal = u_model.predict(base_features)[0]
    B20 = b20_model.predict(base_features)[0]

    # =========================
    # FINAL FEATURES (NO gz)
    # =========================
    tau_features = np.array([
        CN, A1, A2, BA,
        E[0], E[1], E[2], E[3],
        B20, Ucal
    ]).reshape(1, -1)

    # =========================
    # FINAL PREDICTION
    # =========================
    log_tau = tau_model.predict(tau_features)[0]

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Predictions")

    st.success(f"Ucal: {Ucal:.2f}")
    st.success(f"B20: {B20:.4f}")
    st.success(f"log(tau0): {log_tau:.4f}")
