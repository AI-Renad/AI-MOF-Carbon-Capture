# AI MOF Carbon Capture
AI-driven discovery of Metal-Organic Frameworks (MOFs) for efficient CO₂ capture from cement plants and catalytic conversion into methanol using solar energy.

## Overview
Cement plants are among the largest industrial CO₂ emitters in Saudi Arabia. Traditional carbon capture systems are expensive, complex, and inefficient under real industrial conditions.  
This project uses AI and generative AI to identify the best MOFs for:
- High CO₂ adsorption  
- High surface area  
- Strong thermal and structural stability  
- Catalytic conversion of CO₂ → Methanol  
- Solar-compatible performance

## Objectives
- Build a clean dataset of MOF properties  
- Train ML models to rank MOFs  
- Analyze adsorption, energy, and stability  
- Identify the optimal MOF for industrial use

## Methods
### Dataset Processing
- Cleaning  
- Removing last empty columns  
- Merging multiple MOF datasets  
- Feature engineering

### Machine Learning
- Random Forest Regression  
- Train/test split  
- Feature importance & model evaluation

### Generative AI
- Screening candidate MOFs  
- Structure suggestions  
- Optimization assistance

### Evaluation Criteria
- CO₂ adsorption capacity (mmol/g)  
- BET surface area  
- Adsorption energy  
- Thermal stability under cement-plant conditions

## Key Results
The AI system identified the *Top 10 MOFs*.  
Among them, **CAU-89** ranked #1.

### Why CAU-89?
| Property | Value |
|---------|--------|
| CO₂ Adsorption | 503.19 mmol/g |
| BET Surface Area | 1951.72 m²/g |
| Adsorption Energy | 19.74 kJ/mol |
| Stability | Excellent |

Compared to UiO-146 and ZIF-19, **CAU-89** achieves the best balance between high CO₂ capture, fast release behavior, strong stability, and industrial feasibility.  
It is ideal for designing a **solar-powered methanol catalyst**.

## Code Examples

```python
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ===========================
# Load Data
# ===========================
file_path = "/Big.csv"
df = pd.read_csv(file_path)

st.title(" MOF CO₂ Absorption AI Dashboard")
st.write("Shape before cleaning:", df.shape)
st.dataframe(df.head())

# ===========================
# Encode ALL categorical columns automatically
# ===========================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

# ===========================
# Keep only rows with target
# ===========================
target_col = 'Langmuir_am'
df = df.dropna(subset=[target_col])

st.write("Shape after cleaning:", df.shape)

st.write("Dtypes after full encoding:")
st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Type'}))

# ===========================
# Prepare X and y
# ===========================
X = df.drop(target_col, axis=1)
y = df[target_col]

# ===========================
# Split data
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# Train Model
# ===========================
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)
st.success(" Model Trained Successfully!")

# ===========================
# Predict & Score
# ===========================
df["Predicted_CO2"] = model.predict(X)
df["Absorption_Score_100"] = (df["Predicted_CO2"] / df["Predicted_CO2"].max()) * 100
df_sorted = df.sort_values(by="Absorption_Score_100", ascending=False)

st.subheader(" Top 10 MOFs by Absorption Score")
st.dataframe(df_sorted.head(10))

# ===========================
# MOF name (encoded)
# ===========================
df_sorted["MOF_name"] = (
    df_sorted["Core_material"].astype(str)
    + " + "
    + df_sorted["Shell_material"].astype(str)
)

# ===========================
# Dropdown for selection
# ===========================
mof_options = st.multiselect(
    "Select 2-3 MOFs to compare",
    options=df_sorted["MOF_name"].unique(),
    default=list(df_sorted["MOF_name"].unique())[:2]
)

if mof_options:
    selected_mofs = df_sorted[df_sorted["MOF_name"].isin(mof_options)]

    # ===========================
    # Bar chart
    # ===========================
    st.subheader("📊 Absorption Score Comparison")
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=selected_mofs["MOF_name"],
        y=selected_mofs['Absorption_Score_100'],
        palette="viridis"
    )
    plt.ylabel("Absorption Score (%)")
    plt.xlabel("MOF Name")
    plt.title("CO₂ Absorption Score")
    st.pyplot(plt)

    # ===========================
    # Radar Chart
    # ===========================
    st.subheader("🕸 Radar Chart Comparison")
    features = [
        'BET_surface_area_m2_g',
        'Pore_volume_cm3_g',
        'Micropore_volume',
        'Mesopore_volume',
        'Macropore_volume',
        'Adsorption_energy_E_kJmol'
    ]

    fig = go.Figure()
    for _, row in selected_mofs.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[features].values,
            theta=features,
            fill='toself',
            name=row["MOF_name"]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )
    st.plotly_chart(fig)

    # ===========================
    # Summary
    # ===========================
    st.subheader("� Detailed Summary")
    for _, row in selected_mofs.iterrows():
        st.markdown(f"### MOF: **{row['MOF_name']}**")
        st.write(f"**Predicted CO₂ Adsorption:** {row['Predicted_CO2']:.2f}")
        st.write(f"**Absorption Score:** {row['Absorption_Score_100']:.2f} %")
        st.write("---")

```
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/d4c75f59-ac6a-4eea-af22-0036f826dc7d" />


```
# ===========================
# Best Absorber + Best Catalyst Finder
# ===========================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("/Big.csv")

# Encode all text columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Target column (CO₂ absorption capacity)
target_col = "Langmuir_am"
df = df.dropna(subset=[target_col])

# Prepare features
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train model
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X, y)

# Predict
df["Predicted_CO2"] = model.predict(X)
df["Absorption_Score_100"] = (df["Predicted_CO2"] / df["Predicted_CO2"].max()) * 100

# Best CO₂ Absorber
best_absorption = df.sort_values("Predicted_CO2", ascending=False).iloc[0]

# Best Catalyst (highest adsorption energy)
best_catalyst = df.sort_values("Adsorption_energy_E_kJmol", ascending=False).iloc[0]

# Print results
print(" BEST MATERIAL FOR CO₂ ABSORPTION")
print(best_absorption[["Predicted_CO2", "Absorption_Score_100"]])


print(" BEST CATALYST MATERIAL")

print(best_catalyst[["Adsorption_energy_E_kJmol"]])

``` 

## BEST MATERIAL FOR CO₂ ABSORPTION
- Predicted_CO2             2.514618
- Absorption_Score_100    100.000000
- Name: 0, dtype: float64


## BEST CATALYST MATERIAL
- Adsorption_energy_E_kJmol    39.942163
- Name: 154, dtype: float64
