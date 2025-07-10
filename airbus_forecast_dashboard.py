import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ----------------------
# Build a mock model (same logic as training)
# ----------------------
model = Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier())
])

# Train with synthetic data for demo
np.random.seed(42)
n = 100
X_train = pd.DataFrame({
    'PriorPresence': np.random.randint(0, 2, n),
    'MoUs': np.random.randint(0, 5, n),
    'IncentiveSARm': np.random.uniform(0, 600, n),
    'MarketPull': np.random.uniform(0, 100, n),
    'ComponentCostUSD': np.random.uniform(500, 2500, n),
    'EcosystemReady': np.random.randint(0, 3, n),
    'PeerActivity': np.random.randint(0, 4, n)
})

y_train = ((0.2
            + 0.15 * X_train['PriorPresence']
            + 0.08 * X_train['MoUs']
            + 0.001 * X_train['IncentiveSARm']
            + 0.002 * X_train['MarketPull']
            - 0.0001 * X_train['ComponentCostUSD']
            + 0.1 * X_train['EcosystemReady']
            + 0.05 * X_train['PeerActivity']) > 0.5).astype(int)

model.fit(X_train, y_train)

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="Airbus Localization Forecast", layout="centered")
st.title("📡 Airbus Namaat Localization Forecast")
st.markdown("""
Use this interactive tool to estimate the probability that Airbus will establish a secure-handset manufacturing facility in Saudi Arabia under the Namaat program.
""")

# Inputs
st.sidebar.header("📊 Forecast Inputs")
prior = st.sidebar.selectbox("Does Airbus currently operate in KSA?", ["No", "Yes"])
mous = st.sidebar.slider("MoUs signed with local stakeholders", 0, 5, 2)
incentive = st.sidebar.slider("Government incentive (SAR million)", 0, 600, 250)
market = st.sidebar.slider("Strategic market pull (0-100)", 0, 100, 70)
cost = st.sidebar.slider("Unit cost of hardware (USD)", 500, 2500, 1575)
ecosystem = st.sidebar.slider("Ecosystem readiness (0=none, 2=fully ready)", 0, 2, 1)
peers = st.sidebar.slider("# of peer companies localizing", 0, 5, 3)

# Create input DataFrame
input_df = pd.DataFrame([{
    'PriorPresence': 1 if prior == "Yes" else 0,
    'MoUs': mous,
    'IncentiveSARm': incentive,
    'MarketPull': market,
    'ComponentCostUSD': cost,
    'EcosystemReady': ecosystem,
    'PeerActivity': peers
}])

# Predict
prob = model.predict_proba(input_df)[0, 1]
st.metric("📈 Probability Airbus Localizes (within 3 years)", f"{prob:.2%}")

# Show trend chart
months = np.arange(0, 24)
monthly_growth = prob * (1 - np.exp(-0.15 * months)) / (1 - np.exp(-0.15 * 23))
fig, ax = plt.subplots()
ax.plot(months, monthly_growth, marker='o')
ax.set_title("Forecasted Monthly Localization Probability")
ax.set_xlabel("Months Ahead")
ax.set_ylabel("Probability")
ax.grid(True)
ax.set_ylim(0, 1)
st.pyplot(fig)

st.caption("Model is illustrative and based on simulated historical data.")
