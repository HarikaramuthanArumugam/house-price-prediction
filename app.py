import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #111827 0%, #1e1b4b 50%, #172554 100%);
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    color: #c4b5fd !important;
    font-size: 44px !important;
}

h2, h3 {
    color: #a5b4fc !important;
}

p, label, .stCaption {
    color: #e5e7eb !important;
}

.stMarkdown {
    color: #e5e7eb;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #7c3aed, #ec4899);
    color: white !important;
    font-weight: 700;
    height: 50px;
    font-size: 16px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9, #db2777);
    color: white !important;
}

[data-testid="stMetric"] {
    background: #1e293b;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #475569;
}

[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stAlert"] {
    color: #e5e7eb !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

data = pd.read_csv("train.csv")

data["Bathrooms"] = data["FullBath"] + (data["HalfBath"] * 0.5)

x = data[["GrLivArea", "BedroomAbvGr", "Bathrooms"]]
y = data["SalePrice"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ---------------- HEADER ----------------

st.title("🏠 House Price Predictor")

st.write(
    "Estimate the sale price of a house using square footage, "
    "bedrooms and bathrooms."
)

st.divider()


# ---------------- INPUT + PREDICTION ----------------

left, right = st.columns([1, 1], gap="large")


with left:

    st.subheader("🏡 Property Details")

    st.caption("Adjust the values below and generate a prediction.")

    area = st.slider(
        "📐 Living Area (sq ft)",
        min_value=500,
        max_value=5000,
        value=1500,
        step=50
    )

    bedrooms = st.slider(
        "🛏️ Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.slider(
        "🛁 Bathrooms",
        min_value=1.0,
        max_value=8.0,
        value=2.0,
        step=0.5
    )

    st.write("")

    predict = st.button("✨ Predict House Price")


with right:

    st.subheader("💰 Prediction")

    if predict:

        new_house = pd.DataFrame({
            "GrLivArea": [area],
            "BedroomAbvGr": [bedrooms],
            "Bathrooms": [bathrooms]
        })

        price = model.predict(new_house)[0]

        st.success("Prediction generated successfully!")

        st.metric(
            "Estimated Sale Price",
            f"${price:,.2f}"
        )

        st.caption(
            f"{area:,} sq ft  •  "
            f"{bedrooms} bedrooms  •  "
            f"{bathrooms} bathrooms"
        )

    else:

        st.info(
            "Enter the property details on the left "
            "and click **Predict House Price**."
        )


st.divider()


# ---------------- MODEL PERFORMANCE ----------------

st.subheader("📊 Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "R² Score",
        f"{r2:.2%}"
    )

with m2:
    st.metric(
        "Mean Absolute Error",
        f"${mae:,.0f}"
    )

with m3:
    st.metric(
        "Training Houses",
        f"{len(x_train):,}"
    )


st.divider()


# ---------------- ABOUT THE MODEL ----------------

st.subheader("🧠 About This Model")

a, b, c = st.columns(3)

with a:
    st.markdown("### 📐 Area")
    st.write(
        "The model considers the above-ground living area "
        "of the house."
    )

with b:
    st.markdown("### 🛏️ Bedrooms")
    st.write(
        "The number of bedrooms is used as one of the "
        "main prediction features."
    )

with c:
    st.markdown("### 🛁 Bathrooms")
    st.write(
        "Full and half bathrooms are combined into a "
        "single bathroom feature."
    )


st.divider()

st.caption("Linear Regression | Python | Pandas | Scikit-learn | Streamlit")