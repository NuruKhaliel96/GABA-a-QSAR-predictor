import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Load the trained Stacking Regressor
@st.cache_resource
def load_model():
    return joblib.load('gaba_qsar_model.pkl')

model = load_model()

# 2. Build the Web Interface
st.title(" GABA-A Ligand Potency Predictor")
st.markdown("""
This machine learning application predicts the $pIC_{50}$ binding affinity of compounds for the GABA-A receptor. 
It utilizes a Stacking Regressor (Random Forest + Support Vector Regression, synthesized via Ridge Regression) trained on ChEMBL bioactivity data.
""")

# Default SMILES is Diazepam (Valium) - a known GABA-A modulator
smiles_input = st.text_input("Enter a SMILES string to predict potency:", "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13")

if st.button("Predict $pIC_{50}$"):
    # 3. Process the Input and Predict
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol:
        # Generate 2048-bit Morgan Fingerprint (radius 2)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)
        
        # Run inference
        prediction = model.predict(fp_array)[0]
        
        st.success(f"**Predicted $pIC_{50}$ Potency:** {prediction:.3f}")
        st.info("Higher $pIC_{50}$ values indicate greater potency.")
    else:
        st.error("Invalid SMILES string. Please check your chemical structure and try again.")
