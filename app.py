import streamlit as st
import numpy as np
import joblib

from math import acos, degrees
from itertools import combinations

# =========================================================
# CONSTANTS
# =========================================================
CM_TO_K = 1.44

# =========================================================
# MODEL PATHS
# =========================================================
MODEL_PATHS = {

    "D4h": {
        "Ucal": "D4h_U_GB_model.joblib",
        "B20": "D4h_B20_GB_model.joblib"
    },

    "D5h": {
        "Ucal": "D5h_U_GB_model.joblib",
        "B20": "D5h_B20_GB_model.joblib"
    },

    "D6h": {
        "Ucal": "D6h_U_GB_model.joblib",
        "B20": "D6h_B20_GB_model.joblib"
    }
}

UEFF_MODEL = "Ueff_GB_model.joblib"
TAU_MODEL  = "tio_GB_model.joblib"

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="Dy Magnetic Property Predictor",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models(sym):

    return (
        joblib.load(MODEL_PATHS[sym]["Ucal"]),
        joblib.load(MODEL_PATHS[sym]["B20"]),
        joblib.load(UEFF_MODEL),
        joblib.load(TAU_MODEL),
    )

# =========================================================
# READ XYZ
# =========================================================
def read_xyz(file):

    lines = file.read().decode().splitlines()

    atoms = []
    coords = []

    for line in lines:

        p = line.split()

        if len(p) < 4:
            continue

        try:

            atoms.append(p[0])

            coords.append(
                list(map(float, p[1:4]))
            )

        except:
            continue

    return atoms, np.array(coords)

# =========================================================
# FIND DY
# =========================================================
def find_dy(atoms):

    for i, a in enumerate(atoms):

        if str(a).lower() == "dy" or str(a) == "66":
            return i

    return None

# =========================================================
# DISTANCE
# =========================================================
def dist(a, b):

    return np.linalg.norm(a - b)

# =========================================================
# ANGLE
# =========================================================
def angle(a, b, c):

    ba = a - b
    bc = c - b

    return degrees(
        acos(
            np.clip(
                np.dot(ba, bc) /
                (
                    np.linalg.norm(ba)
                    *
                    np.linalg.norm(bc)
                ),
                -1,
                1
            )
        )
    )

# =========================================================
# FIRST COORDINATION SPHERE
# =========================================================
def get_first_coordination_sphere(
        atoms,
        coords,
        dy_idx,
        total_atoms):

    Dy = coords[dy_idx]

    candidates = []

    for i, coord in enumerate(coords):

        if i == dy_idx:
            continue

        atom = atoms[i]

        # skip H
        if atom.upper() == "H":
            continue

        d = dist(Dy, coord)

        if 1.7 <= d <= 3.6:

            candidates.append(
                (i, atom, d)
            )

    candidates = sorted(
        candidates,
        key=lambda x: x[2]
    )

    selected = candidates[:total_atoms]

    return selected

# =========================================================
# PLANE DEVIATION
# =========================================================
def plane_deviation(points):

    centroid = np.mean(points, axis=0)

    shifted = points - centroid

    _, _, vh = np.linalg.svd(shifted)

    normal = vh[-1]

    distances = np.abs(
        np.dot(shifted, normal)
    )

    return np.mean(distances)

# =========================================================
# FIND AXIAL ATOMS
# =========================================================
def find_axial_atoms(
        symmetry,
        atoms,
        coords,
        dy_idx,
        sphere):

    eq_num = {
        "D4h": 4,
        "D5h": 5,
        "D6h": 6
    }[symmetry]

    indices = [x[0] for x in sphere]

    best_plane = None

    best_score = 999999

    # =====================================================
    # TEST ALL POSSIBLE EQUATORIAL PLANES
    # =====================================================
    for subset in combinations(indices, eq_num):

        pts = np.array([
            coords[i]
            for i in subset
        ])

        # planarity
        dev = plane_deviation(pts)

        # remaining atoms = axial
        axial = [
            i for i in indices
            if i not in subset
        ]

        ax1, ax2 = axial

        Dy = coords[dy_idx]

        d1 = dist(Dy, coords[ax1])
        d2 = dist(Dy, coords[ax2])

        avg_axial = (d1 + d2) / 2

        # =================================================
        # D4h SPECIAL RULE
        # =================================================
        if symmetry == "D4h":

            # prioritize shortest axial bonds
            score = (
                dev * 100
                +
                avg_axial
            )

        # =================================================
        # D5h / D6h
        # =================================================
        else:

            score = dev

        if score < best_score:

            best_score = score

            best_plane = subset

            best_axial = axial

    ax1, ax2 = best_axial

    return ax1, ax2, best_plane

# =========================================================
# EQUATORIAL ATOMS
# =========================================================
def get_equatorial_atoms(
        sphere,
        eq_plane):

    eq_atoms = []

    for item in sphere:

        idx, atom, d = item

        if idx in eq_plane:

            eq_atoms.append(
                (idx, atom, d)
            )

    eq_atoms = sorted(
        eq_atoms,
        key=lambda x: x[2]
    )

    return eq_atoms

# =========================================================
# UNCERTAINTY
# =========================================================
def get_uncertainty(model, X):

    try:

        preds = np.array([
            tree.predict(X)[0]
            for tree in model.estimators_.ravel()
        ])

        return np.std(preds)

    except:

        return 0.0

# =========================================================
# UI
# =========================================================
st.title(
    "Dy Magnetic Property Predictor"
)

xyz_file = st.file_uploader(
    "Upload XYZ file",
    type=["xyz"]
)

symmetry = st.selectbox(
    "Select symmetry",
    ["D4h", "D5h", "D6h"]
)

# =========================================================
# RUN
# =========================================================
if st.button("Predict"):

    if xyz_file is None:

        st.error(
            "Please upload XYZ file"
        )

        st.stop()

    atoms, coords = read_xyz(xyz_file)

    dy_idx = find_dy(atoms)

    if dy_idx is None:

        st.error("Dy atom not found")

        st.stop()

    # =====================================================
    # TOTAL COORDINATION
    # =====================================================
    TOTAL_COORD = {

        "D4h": 6,
        "D5h": 7,
        "D6h": 8

    }[symmetry]

    # =====================================================
    # FIRST COORDINATION SPHERE
    # =====================================================
    sphere = get_first_coordination_sphere(
        atoms,
        coords,
        dy_idx,
        TOTAL_COORD
    )

    if len(sphere) < TOTAL_COORD:

        st.error(
            f"Only {len(sphere)} "
            f"coordinating atoms found"
        )

        st.stop()

    # =====================================================
    # FIND AXIAL
    # =====================================================
    ax1, ax2, eq_plane = find_axial_atoms(
        symmetry,
        atoms,
        coords,
        dy_idx,
        sphere
    )

    # =====================================================
    # EQUATORIAL
    # =====================================================
    eq_atoms = get_equatorial_atoms(
        sphere,
        eq_plane
    )

    # =====================================================
    # AXIAL PARAMETERS
    # =====================================================
    Dy = coords[dy_idx]

    Ax1 = coords[ax1]
    Ax2 = coords[ax2]

    A1 = dist(Dy, Ax1)
    A2 = dist(Dy, Ax2)

    if A1 > A2:
        A1, A2 = A2, A1

    theta = angle(
        Ax1,
        Dy,
        Ax2
    )

    BA_model = 180 - theta

    BA_display = theta

    # =====================================================
    # EQUATORIAL DISTANCES
    # =====================================================
    eq_distances = sorted([
        x[2]
        for x in eq_atoms
    ])

    CN = {

        "D4h": 4,
        "D5h": 5,
        "D6h": 6

    }[symmetry]

    if len(eq_distances) != CN:

        st.error(
            f"Expected {CN} equatorial atoms "
            f"found {len(eq_distances)}"
        )

        st.stop()

    
    # =====================================================
    # STRUCTURAL PARAMETERS
    # =====================================================
    st.subheader(
        "Structural Parameters"
    )

    col1, col2 = st.columns(2)

    # =====================================================
    # AXIAL
    # =====================================================
    with col1:

        st.markdown("### Axial")

        st.write(f"A1: {A1:.3f} Å")

        st.write(f"A2: {A2:.3f} Å")

        st.write(
            f"BA: {BA_display:.2f}°"
        )

    # =====================================================
    # EQUATORIAL
    # =====================================================
    with col2:

        st.markdown("### Equatorial")

        for i, val in enumerate(eq_distances):

            st.write(
                f"E{i+1}: "
                f"{val:.3f} Å"
            )

    # =====================================================
    # LOAD MODELS
    # =====================================================
    u_model, b20_model, ueff_model, tau_model = \
        load_models(symmetry)

    # =====================================================
    # FEATURES
    # =====================================================
    base_features = np.array([

        A1,
        A2,

        BA_model,

        *eq_distances

    ]).reshape(1, -1)

    # =====================================================
    # PREDICTIONS
    # =====================================================
    Ucal = u_model.predict(
        base_features
    )[0]

    B20 = b20_model.predict(
        base_features
    )[0]

    final_features = np.array([

        CN,

        A1,
        A2,

        BA_model,

        eq_distances[0],
        eq_distances[1],
        eq_distances[2],
        eq_distances[3],

        B20,
        Ucal

    ]).reshape(1, -1)

    Ueff = ueff_model.predict(
        final_features
    )[0]

    log_tau = tau_model.predict(
        final_features
    )[0]

    # =====================================================
    # UNCERTAINTY
    # =====================================================
    Ucal_err_cm = get_uncertainty(
        u_model,
        base_features
    )

    Ueff_err_cm = get_uncertainty(
        ueff_model,
        final_features
    )

    logtau_err = get_uncertainty(
        tau_model,
        final_features
    )

    # =====================================================
    # CONVERSION
    # =====================================================
    Ucal_K = Ucal * CM_TO_K

    Ueff_K = Ueff * CM_TO_K

    Ucal_err_K = Ucal_err_cm * CM_TO_K

    Ueff_err_K = Ueff_err_cm * CM_TO_K
    # =====================================================
    # ORBACH TEMPERATURE (Tor)
    # =====================================================
    TAU_REF = 100.0  # seconds

    # Convert predicted log10(tau0) to tau0 (s)
    tau0 = 10 ** log_tau

    # Natural logarithm term
    ln_term = np.log(tau0 / TAU_REF)

    # Tor using Ueff in Kelvin
    Tor = -Ueff_K / ln_term

    # =====================================================
    # Uncertainty (using Ueff in Kelvin)
    # =====================================================
    dT_dU = -1.0 / ln_term

    dT_dlogtau = (
        Ueff_K * np.log(10)
        / (ln_term ** 2)
    )

    Tor_err = np.sqrt(
        (dT_dU * Ueff_err_K) ** 2 +
        (dT_dlogtau * logtau_err) ** 2
    )
    # =====================================================
    # OUTPUT
    # =====================================================
    st.subheader("Predictions")

    st.success(
        f"Ucal: "
        f"{Ucal:.2f} ± {Ucal_err_cm:.2f} cm⁻¹   "
        f"({Ucal_K:.2f} ± {Ucal_err_K:.2f} K)"
    )

    st.success(
        f"B20: {B20:.4f}"
    )

    st.success(
        f"Ueff: "
        f"{Ueff:.2f} ± {Ueff_err_cm:.2f} cm⁻¹   "
        f"({Ueff_K:.2f} ± {Ueff_err_K:.2f} K)"
    )

    st.success(
        f"log(τ₀): "
        f"{log_tau:.4f} ± {logtau_err:.4f}"
    )
    st.success(
        f"Tor (τref = 100 s): "
        f"{Tor:.2f} ± {Tor_err:.2f} K"
    )
    st.markdown(
        r"""
    <div style="text-align: left; font-size: 22px;">
    $$
    T_{\mathrm{Or}}
    =
    -\frac{U_{\mathrm{eff}}}
    {\ln\left(\frac{\tau_{0}}{\tau_{\mathrm{ref}}}\right)}
    $$
    </div>
    """,
        unsafe_allow_html=True,
    )
