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
    },
}

UEFF_MODEL = "Ueff_GB_model.joblib"
TAU_MODEL  = "tio_GB_model.joblib"

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Dy Magnetic Property Predictor",
                   layout="wide")

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

    atoms = []
    coords = []

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

# =========================
# FIND DY
# =========================
def find_dy(atoms):

    for i, a in enumerate(atoms):

        if str(a).lower() == "dy" or str(a) == "66":
            return i

    return None

# =========================
# DISTANCE
# =========================
def dist(a, b):

    return np.linalg.norm(a - b)

# =========================
# ANGLE
# =========================
def angle(a, b, c):

    ba = a - b
    bc = c - b

    return degrees(
        acos(
            np.clip(
                np.dot(ba, bc) /
                (np.linalg.norm(ba) * np.linalg.norm(bc)),
                -1,
                1
            )
        )
    )

# =========================
# FIRST COORDINATION SPHERE
# =========================
def get_first_coordination_sphere(atoms,
                                  coords,
                                  dy_idx,
                                  total_atoms):

    Dy = coords[dy_idx]

    candidates = []

    for i, coord in enumerate(coords):

        if i == dy_idx:
            continue

        atom = atoms[i]

        # skip hydrogen
        if atom.upper() == "H":
            continue

        d = dist(Dy, coord)

        # coordination cutoff
        if 1.7 <= d <= 3.6:

            candidates.append((i, atom, d))

    # sort by distance
    candidates = sorted(candidates,
                        key=lambda x: x[2])

    # select closest atoms
    selected = candidates[:total_atoms]

    return selected

# =========================
# FIND Cn AXIS + AXIAL ATOMS
# =========================
def find_axial_atoms(symmetry,
                     atoms,
                     coords,
                     dy_idx,
                     sphere):

    Dy = coords[dy_idx]

    indices = [x[0] for x in sphere]

    vectors = []

    for idx in indices:

        vec = coords[idx] - Dy

        vectors.append(vec)

    vectors = np.array(vectors)

    # principal axis
    _, _, vh = np.linalg.svd(vectors)

    axis = vh[0]

    # projections along axis
    projections = np.dot(vectors, axis)

    # most positive
    pos_idx = np.argmax(projections)

    # most negative
    neg_idx = np.argmin(projections)

    ax1 = indices[pos_idx]
    ax2 = indices[neg_idx]

    return ax1, ax2

# =========================
# EQUATORIAL ATOMS
# =========================
def get_equatorial_atoms(sphere,
                         ax1,
                         ax2):

    eq_atoms = []

    for item in sphere:

        idx, atom, d = item

        if idx in [ax1, ax2]:
            continue

        eq_atoms.append((idx, atom, d))

    eq_atoms = sorted(eq_atoms,
                      key=lambda x: x[2])

    return eq_atoms

# =========================
# UNCERTAINTY
# =========================
def get_uncertainty(model, X):

    try:

        preds = np.array([
            tree.predict(X)[0]
            for tree in model.estimators_.ravel()
        ])

        return np.std(preds)

    except:

        return 0.0

# =========================
# UI
# =========================
st.title("Dy Magnetic Property Predictor")

xyz_file = st.file_uploader(
    "Upload XYZ file",
    type=["xyz"]
)

symmetry = st.selectbox(
    "Select symmetry",
    ["D4h", "D5h", "D6h"]
)

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

    # =========================
    # TOTAL COORDINATION NUMBER
    # =========================
    TOTAL_COORD = {
        "D4h": 6,
        "D5h": 7,
        "D6h": 8
    }[symmetry]

    # =========================
    # FIRST COORDINATION SPHERE
    # =========================
    sphere = get_first_coordination_sphere(
        atoms,
        coords,
        dy_idx,
        TOTAL_COORD
    )

    if len(sphere) < TOTAL_COORD:

        st.error(
            f"Only {len(sphere)} coordinating atoms found"
        )

        st.stop()

    # =========================
    # FIND AXIAL ATOMS
    # =========================
    ax1, ax2 = find_axial_atoms(
        symmetry,
        atoms,
        coords,
        dy_idx,
        sphere
    )

    Dy = coords[dy_idx]

    Ax1 = coords[ax1]
    Ax2 = coords[ax2]

    # =========================
    # AXIAL PARAMETERS
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
    eq_atoms = get_equatorial_atoms(
        sphere,
        ax1,
        ax2
    )

    eq_distances = sorted([
        x[2] for x in eq_atoms
    ])

    CN = {
        "D4h": 4,
        "D5h": 5,
        "D6h": 6
    }[symmetry]

    if len(eq_distances) != CN:

        st.error(
            f"Expected {CN} equatorial atoms, "
            f"found {len(eq_distances)}"
        )

        st.stop()

    # =========================
    # DISPLAY STRUCTURE INFO
    # =========================
    st.subheader("Detected Coordination Sphere")

    st.write("### Axial Atoms")

    st.write(
        f"Atom 1: Index {ax1+1} "
        f"({atoms[ax1]})"
    )

    st.write(
        f"Atom 2: Index {ax2+1} "
        f"({atoms[ax2]})"
    )

    st.write("### Equatorial Atoms")

    for item in eq_atoms:

        idx, atom, d = item

        st.write(
            f"Index {idx+1} "
            f"({atom})   "
            f"{d:.3f} Å"
        )

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

            st.write(
                f"E{i+1}: {val:.3f} Å"
            )

    # =========================
    # LOAD MODELS
    # =========================
    u_model, b20_model, ueff_model, tau_model = \
        load_models(symmetry)

    # =========================
    # MODEL FEATURES
    # =========================
    base_features = np.array([
        A1,
        A2,
        BA_model,
        *eq_distances
    ]).reshape(1, -1)

    # =========================
    # PREDICTIONS
    # =========================
    Ucal = u_model.predict(base_features)[0]

    B20 = b20_model.predict(base_features)[0]

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

    Ueff = ueff_model.predict(final_features)[0]

    log_tau = tau_model.predict(final_features)[0]

    # =========================
    # UNCERTAINTY
    # =========================
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

    # =========================
    # CONVERSION
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
