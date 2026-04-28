import streamlit as st
import numpy as np
import joblib
from math import acos, degrees

# =========================
# CONSTANTS
# =========================
CM_TO_K = 1.44

# =========================
# MODEL PATHS
# =========================
MODEL_PATHS = {
    "D4h": {"Ucal": "D4h_U_GB_model.joblib", "B20": "D4h_B20_GB_model.joblib"},
    "D5h": {"Ucal": "D5h_U_GB_model.joblib", "B20": "D5h_B20_GB_model.joblib"},
    "D6h": {"Ucal": "D6h_U_GB_model.joblib", "B20": "D6h_B20_GB_model.joblib"},
}

UEFF_MODEL = "Ueff_GB_model.joblib"
TAU_MODEL  = "tio_GB_model.joblib"

st.set_page_config(page_title="Dy Predictor", layout="wide")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models(sym):
    return (
        joblib.load(MODEL_PATHS[sym]["Ucal"]),
        joblib.load(MODEL_PATHS[sym]["B20"]),
        joblib.load(UEFF_MODEL),
        joblib.load(TAU_MODEL),
    )

# =========================
# FUNCTIONS
# =========================
def read_xyz(file):
    lines = file.read().decode().splitlines()
    atoms, coords = [], []

    for line in lines:
        p = line.split()
        if len(p) < 4:
            continue
        try:
            atoms.append(p[0])
            coords.append(list(map(float, p[1:4])))
        except:
            continue

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
                np.dot(ba, bc) /
                (np.linalg.norm(ba) * np.linalg.norm(bc)),
                -1, 1
            )
        )
    )

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
            candidates.append((i, atom, d))

    return sorted(candidates, key=lambda x: x[2])

# =========================
# NEW: UNCERTAINTY FUNCTION
# =========================
def get_uncertainty(model, X):
    try:
        preds = np.array([
            tree.predict(X)[0] for tree in model.estimators_.ravel()
        ])
        return np.std(preds)
    except:
        return 0.0

# =========================
# UI
# =========================
st.title("Dy Magnetic Property Predictor")

xyz_file = st.file_uploader("Upload XYZ file", type=["xyz"])

col1, col2 = st.columns(2)
ax1_input = col1.number_input("Axial atom index 1", min_value=1)
ax2_input = col2.number_input("Axial atom index 2", min_value=1)

symmetry = st.selectbox("Select symmetry", ["D4h", "D5h", "D6h"])

# =========================
# RUN
# =========================
if st.button("Predict"):

    if xyz_file is None:
        st.error("Please upload XYZ file")
        st.stop()

    atoms, coords = read_xyz(xyz_file)
    dy_idx = find_dy(atoms)

    if dy_idx is None:
        st.error("Dy atom not found")
        st.stop()

    ax1 = int(ax1_input) - 1
    ax2 = int(ax2_input) - 1

    Dy = coords[dy_idx]
    Ax1 = coords[ax1]
    Ax2 = coords[ax2]

    # =========================
    # AXIAL
    # =========================
    A1 = dist(Dy, Ax1)
    A2 = dist(Dy, Ax2)

    if A1 > A2:
        A1, A2 = A2, A1

    theta = angle(Ax1, Dy, Ax2)

    BA_model = 180 - theta
    BA_display = theta

    # =========================
    # EQUATORIAL
    # =========================
    CN = {"D4h": 4, "D5h": 5, "D6h": 6}[symmetry]

    candidates = get_equatorial_atoms(atoms, coords, dy_idx, ax1, ax2)

    if len(candidates) < CN:
        st.error(f"Only {len(candidates)} equatorial atoms found (need {CN})")
        st.stop()

    selected = candidates[:CN]
    eq_distances = sorted([x[2] for x in selected])

    # =========================
    # STRUCTURAL PARAMETERS
    # =========================
    st.subheader("Structural Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Axial")
        st.write(f"A1: {A1:.3f} Å")
        st.write(f"A2: {A2:.3f} Å")
        st.write(f"BA: {BA_display:.2f}°")

    with col2:
        st.markdown("### Equatorial")
        for i, val in enumerate(eq_distances):
            st.write(f"E{i+1}: {val:.3f} Å")

    # =========================
    # LOAD MODELS
    # =========================
    u_model, b20_model, ueff_model, tau_model = load_models(symmetry)

    base_features = np.array([A1, A2, BA_model, *eq_distances]).reshape(1, -1)

    Ucal = u_model.predict(base_features)[0]
    B20  = b20_model.predict(base_features)[0]

    final_features = np.array([
        CN, A1, A2, BA_model,
        eq_distances[0], eq_distances[1], eq_distances[2], eq_distances[3],
        B20, Ucal
    ]).reshape(1, -1)

    Ueff = ueff_model.predict(final_features)[0]
    log_tau = tau_model.predict(final_features)[0]

    # =========================
    # NEW: DYNAMIC ERRORS
    # =========================
    Ucal_err_cm = get_uncertainty(u_model, base_features)
    Ueff_err_cm = get_uncertainty(ueff_model, final_features)
    logtau_err  = get_uncertainty(tau_model, final_features)

    # =========================
    # CONVERSIONS
    # =========================
    Ucal_K = Ucal * CM_TO_K
    Ueff_K = Ueff * CM_TO_K

    Ucal_err_K = Ucal_err_cm * CM_TO_K
    Ueff_err_K = Ueff_err_cm * CM_TO_K

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Predictions")

    st.success(
        f"Ucal: {Ucal:.2f} ± {Ucal_err_cm:.2f} cm⁻¹  "
        f"({Ucal_K:.2f} ± {Ucal_err_K:.2f} K)"
    )

    st.success(f"B20: {B20:.4f}")

    st.success(
        f"Ueff: {Ueff:.2f} ± {Ueff_err_cm:.2f} cm⁻¹  "
        f"({Ueff_K:.2f} ± {Ueff_err_K:.2f} K)"
    )

    st.success(
        f"log(τ₀): {log_tau:.4f} ± {logtau_err:.4f}"
    )
