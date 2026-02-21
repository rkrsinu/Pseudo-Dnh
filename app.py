import streamlit as st
import numpy as np
import pandas as pd
import joblib
from math import acos, degrees

MODEL_PATH = "Ucal_GB_model.joblib"

st.set_page_config(page_title="Dy Pseudo-Dnh Barrier Height Predictor", layout="wide")


# =========================
# File reader
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


# =========================
# Find Dy
# =========================
def find_dy(atoms):
    for i, a in enumerate(atoms):
        if str(a).lower() == "dy" or str(a) == "66":
            return i
    return None


# =========================
# Geometry
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
# UI
# =========================
st.title("Dy Pseudo-Dnh Barrier Height Predictor")

xyz_file = st.file_uploader("Upload XYZ file", type=["xyz"])

col1, col2 = st.columns(2)
ax1_input = col1.number_input("Axial atom index 1 (1-based)", min_value=1, step=1)
ax2_input = col2.number_input("Axial atom index 2 (1-based)", min_value=1, step=1)

symmetry = st.selectbox("Select symmetry", ["D4h", "D5h", "D6h"])


# =========================
# RUN
# =========================
if st.button("Predict Barrier Height"):

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

    # ================= AXIAL =================
    A1 = dist(Dy, Ax1)
    A2 = dist(Dy, Ax2)

    axial_angle = angle(Ax1, Dy, Ax2)
    BA = abs(180 - axial_angle)

    # ================= EQUATORIAL SELECTION =================
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
            candidates.append((i + 1, d))  # store 1-based index

    CN = {"D4h": 4, "D5h": 5, "D6h": 6}[symmetry]

    if len(candidates) < CN:
        st.error(f"Only {len(candidates)} equatorial atoms found, but {CN} required.")
        st.stop()

    # take shortest N
    candidates = sorted(candidates, key=lambda x: x[1])[:CN]

    eq_distances = sorted([d for _, d in candidates])

    axial_sorted = sorted([A1, A2])

    # ================= RAW STRUCTURAL PARAMETERS =================
    raw_data = [CN, axial_sorted[0], axial_sorted[1], BA] + eq_distances
    raw_columns = (
        ["CN", "A1", "A2", "BA"]
        + [f"BE{i+1}" for i in range(len(eq_distances))]
    )

    raw_df = pd.DataFrame([raw_data], columns=raw_columns)

    st.subheader("All Structural Parameters (Raw)")
    st.dataframe(raw_df, use_container_width=True)

    st.subheader("Selected Equatorial Atom Indices (1-based)")
    st.write([i for i, _ in candidates])

    # ================= AVERAGING FOR ML (HIDDEN) =================
    if symmetry == "D4h":
        eq_model = eq_distances

    elif symmetry == "D5h":
        eq_model = [
            eq_distances[0],
            eq_distances[1],
            np.mean(eq_distances[2:4]),
            eq_distances[4],
        ]

    elif symmetry == "D6h":
        eq_model = [
            eq_distances[0],
            eq_distances[1],
            np.mean(eq_distances[2:5]),
            eq_distances[5],
        ]

    features = np.array(
        [
            CN,
            axial_sorted[0],
            axial_sorted[1],
            BA,
            eq_model[0],
            eq_model[1],
            eq_model[2],
            eq_model[3],
        ]
    ).reshape(1, -1)

    # ================= MODEL =================
    model = joblib.load(MODEL_PATH)

    prediction = model.predict(features)[0]

    st.subheader("Predicted Barrier Height")
    st.success(f"{prediction:.2f}")