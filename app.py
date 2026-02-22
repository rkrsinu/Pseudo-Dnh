import streamlit as st
import numpy as np
import pandas as pd
import joblib
from math import acos, degrees

U_MODEL = "Ucal_GB_model.joblib"
B20_MODEL = "B20_GB_model.joblib"
TAU_MODEL = "tio_GB_model.joblib"

st.set_page_config(page_title="Dy Pseudo-Dnh Magnetic Predictor", layout="wide")


# =========================
# CACHED MODEL LOADING
# =========================
@st.cache_resource
def load_models():
    return (
        joblib.load(U_MODEL),
        joblib.load(B20_MODEL),
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
# UI
# =========================
st.title("Dy Pseudo-Dnh Magnetic Property Predictor")

xyz_file = st.file_uploader("Upload XYZ file", type=["xyz"])

col1, col2 = st.columns(2)
ax1_input = col1.number_input("Axial atom index 1 (1-based)", min_value=1, step=1)
ax2_input = col2.number_input("Axial atom index 2 (1-based)", min_value=1, step=1)

symmetry = st.selectbox("Select symmetry", ["D4h", "D5h", "D6h"])


# =========================
# RUN
# =========================
if st.button("Predict"):

    if xyz_file is None:
        st.error("Please upload an XYZ file")
        st.stop()

    atoms, coords = read_xyz(xyz_file)
    dy_idx = find_dy(atoms)

    if dy_idx is None:
        st.error("Dy atom not found in structure")
        st.stop()

    ax1 = int(ax1_input) - 1
    ax2 = int(ax2_input) - 1

    Dy = coords[dy_idx]
    Ax1 = coords[ax1]
    Ax2 = coords[ax2]

    A1 = dist(Dy, Ax1)
    A2 = dist(Dy, Ax2)
    BA = abs(180 - angle(Ax1, Dy, Ax2))

    candidates = []

    for i, coord in enumerate(coords):

        if i in [dy_idx, ax1, ax2]:
            continue

        d = dist(Dy, coord)

        if not (1.9 <= d <= 3.6):
            continue

        ang1 = angle(Ax1, Dy, coord)
        ang2 = angle(Ax2, Dy, coord)

        if 60 <= ang1 <= 125 and 60 <= ang2 <= 125:
            candidates.append((i + 1, d))

    CN = {"D4h": 4, "D5h": 5, "D6h": 6}[symmetry]

    if len(candidates) < CN:
        st.error(f"Only {len(candidates)} equatorial atoms found, but {CN} required.")
        st.stop()

    candidates = sorted(candidates, key=lambda x: x[1])[:CN]
    eq_distances = sorted([d for _, d in candidates])
    axial_sorted = sorted([A1, A2])

    # -------- DISPLAY RAW --------
    raw_data = [CN, axial_sorted[0], axial_sorted[1], BA] + eq_distances
    raw_columns = ["CN", "A1", "A2", "BA"] + [f"BE{i+1}" for i in range(len(eq_distances))]
    st.subheader("All Structural Parameters (Raw)")
    st.dataframe(pd.DataFrame([raw_data], columns=raw_columns))

    st.subheader("Selected Equatorial Atom Indices (1-based)")
    st.write([i for i, _ in candidates])

    # -------- AVERAGING --------
    if symmetry == "D4h":
        eq_model = eq_distances
    elif symmetry == "D5h":
        eq_model = [eq_distances[0], eq_distances[1],
                    np.mean(eq_distances[2:4]), eq_distances[4]]
    elif symmetry == "D6h":
        eq_model = [eq_distances[0], eq_distances[1],
                    np.mean(eq_distances[2:5]), eq_distances[5]]

    base_features = np.array([
        CN, axial_sorted[0], axial_sorted[1], BA,
        eq_model[0], eq_model[1], eq_model[2], eq_model[3]
    ]).reshape(1, -1)

    # -------- LOAD MODELS --------
    u_model, b20_model, tau_model = load_models()

    # -------- PREDICTIONS --------
    Ucal = u_model.predict(base_features)[0]
    B20 = b20_model.predict(base_features)[0]

    tau_features = np.array([
        CN, axial_sorted[0], axial_sorted[1], BA,
        eq_model[0], eq_model[1], eq_model[2], eq_model[3],
        Ucal, B20
    ]).reshape(1, -1)

    log_tau = tau_model.predict(tau_features)[0]

    # -------- OUTPUT --------
    st.subheader("Predicted Barrier Height")
    st.success(f"{Ucal:.2f}")

    #st.subheader("Predicted B20")
    #st.success(f"{B20:.6f}")

    st.subheader("Predicted Log(τ₀)")
    st.success(f"{log_tau:.4f}")

