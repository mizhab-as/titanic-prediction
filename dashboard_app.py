import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from scripts.titanic_classification import load_data, preprocess


st.set_page_config(
    page_title="Titanic Survival Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {0: "#6f6f6f", 1: "#efefef"}
PLOTLY_TEMPLATE = "plotly_dark"
HERO_IMAGE_URL = "https://images.unsplash.com/photo-1518391846015-55a9cc003b25?auto=format&fit=crop&w=2400&q=80&sat=-100"


def apply_amoled_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg: #121212;
            --card: #1a1a1a;
            --border: #323232;
            --text: #e9e9e9;
            --muted: #a9a9a9;
            --accent: #d9d9d9;
        }

        .stApp {
            background: radial-gradient(circle at 20% 10%, #252525 0%, #1a1a1a 40%, #121212 100%);
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            padding-top: 1rem;
            max-width: 1350px;
        }

        .hero-wrap {
            position: relative;
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 0.75rem;
            min-height: 240px;
            background: #181818;
        }

        .hero-img {
            width: 100%;
            min-height: 240px;
            object-fit: cover;
            filter: grayscale(100%) contrast(120%) brightness(60%);
        }

        .hero-overlay {
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(0, 0, 0, 0.85) 0%, rgba(0, 0, 0, 0.35) 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 24px 28px;
        }

        .hero-title {
            font-size: 30px;
            font-weight: 800;
            letter-spacing: 0.4px;
            color: #ffffff;
            margin-bottom: 6px;
        }

        .hero-subtitle {
            font-size: 15px;
            color: #d8d8d8;
            max-width: 760px;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }

        .badge-chip {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid #454545;
            color: #ececec;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 12px;
            letter-spacing: 0.3px;
        }

        .pro-card {
            background: linear-gradient(180deg, #232323 0%, #1b1b1b 100%);
            border: 1px solid #3a3a3a;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 8px;
            margin-bottom: 10px;
        }

        .pro-title {
            color: #f3f3f3;
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 6px;
        }

        .pro-text {
            color: #cccccc;
            font-size: 14px;
            line-height: 1.5;
        }

        h1, h2, h3, h4 {
            letter-spacing: 0.2px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1d1d1d 0%, #151515 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, #202020 0%, #181818 100%);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 8px 10px;
        }

        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
            color: #f2f2f2;
        }

        [data-testid="stTabs"] button {
            background: #1b1b1b;
            border: 1px solid #363636;
            border-radius: 10px 10px 0 0;
            color: #d8d8d8;
            margin-right: 4px;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            border-color: #6a6a6a;
            background: #242424;
            color: #ffffff;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        .stSlider > div > div {
            background: #202020 !important;
            border: 1px solid #3a3a3a !important;
            color: #f2f2f2 !important;
            border-radius: 10px !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] > div,
        [data-testid="stSidebar"] div[data-baseweb="textarea"] > div {
            background: #212121 !important;
            border: 1px solid #3a3a3a !important;
        }

        /* Hard lock all select/multiselect internals in sidebar to AMOLED dark */
        [data-testid="stSidebar"] div[data-baseweb="select"] {
            background: #212121 !important;
            color: #f5f5f5 !important;
        }

        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"],
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            background: #212121 !important;
            color: #f5f5f5 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div > div > div,
        [data-testid="stSidebar"] div[data-baseweb="select"] [class*="valueContainer"],
        [data-testid="stSidebar"] div[data-baseweb="select"] [class*="singleValue"],
        [data-testid="stSidebar"] div[data-baseweb="select"] [class*="placeholder"] {
            background: #212121 !important;
            color: #f5f5f5 !important;
            -webkit-text-fill-color: #f5f5f5 !important;
        }

        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] [class*="valueContainer"] *,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [class*="valueContainer"] * {
            color: #101010 !important;
            -webkit-text-fill-color: #101010 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] button,
        [data-testid="stSidebar"] div[data-baseweb="select"] svg {
            background: transparent !important;
            color: #f5f5f5 !important;
            fill: #f5f5f5 !important;
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] input:focus,
        div[data-baseweb="select"] input:hover {
            background: transparent !important;
            color: #f2f2f2 !important;
            -webkit-text-fill-color: #f2f2f2 !important;
        }

        div[data-baseweb="select"] > div > div,
        div[data-baseweb="select"] > div > div > div {
            background: transparent !important;
        }

        div[data-baseweb="select"] span,
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stNumberInput label,
        .stTextInput label {
            color: #e6e6e6 !important;
        }

        div[data-baseweb="select"] *:focus,
        div[data-baseweb="input"] *:focus,
        .stSlider *:focus {
            box-shadow: 0 0 0 1px #f0f0f0 !important;
            border-color: #f0f0f0 !important;
        }

        div[data-baseweb="tag"] {
            background: #151515 !important;
            border: 1px solid #2d2d2d !important;
            color: #f2f2f2 !important;
        }

        [data-testid="stMultiSelect"] div[data-baseweb="tag"],
        [data-testid="stSelectbox"] div[data-baseweb="tag"],
        [data-testid="stSidebar"] div[data-baseweb="tag"] {
            background: #161616 !important;
            border: 1px solid #333333 !important;
            color: #f2f2f2 !important;
        }

        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"],
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="tag"],
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background-color: #2a2a2a !important;
            background-image: none !important;
            border-color: #4a4a4a !important;
            box-shadow: none !important;
            color: #f2f2f2 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"]:hover,
        [data-testid="stSidebar"] [data-baseweb="tag"]:focus,
        [data-testid="stSidebar"] [data-baseweb="tag"]:active,
        [data-testid="stSidebar"] [data-baseweb="tag"]:focus-within,
        [data-testid="stSidebar"] [data-baseweb="tag"][aria-selected="true"] {
            background: #303030 !important;
            border-color: #5a5a5a !important;
            color: #f2f2f2 !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] *:hover,
        [data-testid="stSidebar"] [data-baseweb="tag"] *:focus,
        [data-testid="stSidebar"] [data-baseweb="tag"] *:active {
            color: #f2f2f2 !important;
            -webkit-text-fill-color: #f2f2f2 !important;
            background: transparent !important;
            outline: none !important;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"],
        [data-testid="stSidebar"] [data-baseweb="tag"] * {
            user-select: none !important;
            -webkit-user-select: none !important;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] ::selection,
        [data-testid="stSidebar"] [data-baseweb="tag"] *::selection {
            background: transparent !important;
            color: #f2f2f2 !important;
        }

        div[data-baseweb="tag"] span {
            color: #f2f2f2 !important;
        }

        [data-testid="stMultiSelect"] div[data-baseweb="tag"] span,
        [data-testid="stSelectbox"] div[data-baseweb="tag"] span,
        [data-testid="stSidebar"] div[data-baseweb="tag"] span {
            color: #f2f2f2 !important;
            -webkit-text-fill-color: #f2f2f2 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="tag"] *,
        [data-testid="stSidebar"] div[data-baseweb="tag"] span,
        [data-testid="stSidebar"] div[data-baseweb="tag"] p {
            color: #f2f2f2 !important;
            -webkit-text-fill-color: #f2f2f2 !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] svg,
        [data-testid="stSidebar"] [data-baseweb="tag"] path,
        [data-testid="stSidebar"] [data-baseweb="tag"] button {
            color: #f2f2f2 !important;
            fill: #f2f2f2 !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] button:hover,
        [data-testid="stSidebar"] [data-baseweb="tag"] button:focus,
        [data-testid="stSidebar"] [data-baseweb="tag"] button:active {
            color: #f2f2f2 !important;
            fill: #f2f2f2 !important;
            background: transparent !important;
            outline: none !important;
            box-shadow: none !important;
        }

        div[data-baseweb="tag"] svg,
        div[data-baseweb="tag"] button {
            color: #f2f2f2 !important;
            fill: #f2f2f2 !important;
            background: transparent !important;
        }

        div[data-baseweb="slider"] [role="slider"] {
            background: #f1f1f1 !important;
            border: 1px solid #f8f8f8 !important;
        }

        div[data-baseweb="slider"] > div > div {
            background: #3a3a3a !important;
        }

        [data-testid="stSelectbox"] ul,
        [data-testid="stMultiSelect"] ul {
            background: #1f1f1f !important;
            color: #f2f2f2 !important;
            border: 1px solid #3d3d3d !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] ul,
        [role="listbox"] {
            background: #1f1f1f !important;
            color: #f2f2f2 !important;
            border: 1px solid #3d3d3d !important;
        }

        [data-testid="stSelectbox"] li,
        [data-testid="stMultiSelect"] li {
            color: #f2f2f2 !important;
        }

        [data-testid="stSelectbox"] li[aria-selected="true"],
        [data-testid="stMultiSelect"] li[aria-selected="true"] {
            background: #1f1f1f !important;
        }

        [role="option"][aria-selected="true"],
        [data-baseweb="menu"] [aria-selected="true"] {
            background: #1f1f1f !important;
            color: #f2f2f2 !important;
        }

        [data-testid="stSidebar"] [role="option"][aria-selected="true"],
        [data-testid="stSidebar"] [data-baseweb="menu"] [aria-selected="true"] {
            background: #1f1f1f !important;
            color: #f2f2f2 !important;
        }

        [role="option"][aria-selected="true"] *,
        [data-baseweb="menu"] [aria-selected="true"] * {
            color: #f2f2f2 !important;
            -webkit-text-fill-color: #f2f2f2 !important;
        }

        [data-baseweb="menu"] li:hover,
        [role="option"]:hover {
            background: #171717 !important;
        }

        .stButton > button {
            background: linear-gradient(180deg, #f0f0f0 0%, #cecece 100%);
            color: #111;
            border: none;
            border-radius: 10px;
            font-weight: 700;
        }

        .stButton > button:hover {
            filter: brightness(1.05);
        }

        .stDataFrame, .stTable {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        .stMarkdown, p, label, .stCaption {
            color: #e5e5e5 !important;
        }

        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <img class="hero-img" src="{HERO_IMAGE_URL}" alt="Titanic monochrome visual" />
            <div class="hero-overlay">
                <div class="hero-title">Titanic Survival Intelligence</div>
                <div class="hero-subtitle">
                    Layered dark analytics dashboard with interactive filtering,
                    model evaluation, and passenger-level survival prediction.
                </div>
                <div class="badge-row">
                    <span class="badge-chip">LIVE FILTERING</span>
                    <span class="badge-chip">MODEL COMPARISON</span>
                    <span class="badge-chip">ROC + CONFUSION MATRIX</span>
                    <span class="badge-chip">PASSENGER PREDICTION</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#171717",
        plot_bgcolor="#171717",
        font=dict(color="#f2f2f2"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#3a3a3a", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#3a3a3a", zeroline=False)
    return fig


@st.cache_data
def prepare_data() -> pd.DataFrame:
    raw_df = load_data()
    return preprocess(raw_df)


@st.cache_data
def build_model_data(df: pd.DataFrame):
    model_df = df.copy()
    model_df["sex"] = model_df["sex"].map({"female": 1, "male": 0})
    model_df["embarked"] = model_df["embarked"].map({"C": 0, "Q": 1, "S": 2})
    model_df["title"] = model_df["title"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4})
    model_df["age_group"] = model_df["age_group"].map(
        {"Child": 0, "Teen": 1, "Young_Adult": 2, "Middle_Age": 3, "Senior": 4}
    )

    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "family_size",
        "has_cabin",
        "title",
        "age_group",
    ]

    X = model_df[features].fillna(model_df[features].mean(numeric_only=True))
    y = model_df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, features


@st.cache_resource
def train_models(_df: pd.DataFrame):
    X_train, X_test, y_train, y_test, features = build_model_data(_df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "confusion_matrix": confusion_matrix(y_test, pred),
            "report": classification_report(y_test, pred, output_dict=True),
            "proba": prob,
            "auc": roc_auc_score(y_test, prob),
        }

    return results, X_test, y_test, features


def make_confusion_chart(cm, title: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            colorscale=[[0, "#2a2a2a"], [1, "#f1f1f1"]],
            text=cm,
            texttemplate="%{text}",
            hovertemplate="%{y} | %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=50, b=10))
    return style_plot(fig)


def make_roc_chart(model_results, y_test):
    fig = go.Figure()
    for model_name, payload in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, payload["proba"])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={payload['auc']:.3f})",
                line=dict(width=3, color="#f4f4f4" if "Logistic" in model_name else "#8f8f8f"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="#4c4c4c", width=1, dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return style_plot(fig)


def main():
    apply_amoled_theme()
    render_hero()

    df = prepare_data()
    model_results, _X_test, _y_test, features = train_models(df)

    with st.sidebar:
        st.header("Dashboard Controls")
        st.caption("Tune filters to explore survival patterns across cohorts.")
        selected_sex = st.multiselect(
            "Filter by Sex",
            options=sorted(df["sex"].unique()),
            default=sorted(df["sex"].unique()),
        )
        selected_pclass = st.multiselect(
            "Filter by Pclass",
            options=sorted(df["pclass"].unique()),
            default=sorted(df["pclass"].unique()),
        )
        selected_age_group = st.multiselect(
            "Filter by Age Group",
            options=df["age_group"].astype(str).sort_values().unique().tolist(),
            default=df["age_group"].astype(str).sort_values().unique().tolist(),
        )
        selected_title = st.multiselect(
            "Filter by Title",
            options=sorted(df["title"].unique()),
            default=sorted(df["title"].unique()),
        )
        fare_min, fare_max = st.slider(
            "Fare Range",
            min_value=float(df["fare"].min()),
            max_value=float(df["fare"].max()),
            value=(float(df["fare"].quantile(0.01)), float(df["fare"].quantile(0.95))),
        )
        fam_min, fam_max = st.slider(
            "Family Size Range",
            min_value=int(df["family_size"].min()),
            max_value=int(df["family_size"].max()),
            value=(int(df["family_size"].min()), int(df["family_size"].max())),
        )
        survival_focus = st.selectbox("Survival Focus", ["All", "Survived", "Did Not Survive"], index=0)
        st.divider()
        st.caption("Filter status updates in main panel KPIs")

    filtered = df[
        df["sex"].isin(selected_sex)
        & df["pclass"].isin(selected_pclass)
        & df["age_group"].astype(str).isin(selected_age_group)
        & df["title"].isin(selected_title)
        & df["fare"].between(fare_min, fare_max)
        & df["family_size"].between(fam_min, fam_max)
    ]

    if survival_focus == "Survived":
        filtered = filtered[filtered["survived"] == 1]
    elif survival_focus == "Did Not Survive":
        filtered = filtered[filtered["survived"] == 0]

    if filtered.empty:
        st.warning("No rows match the current filters. Adjust filters in the sidebar.")
        return

    overall_survival = df["survived"].mean() * 100
    filtered_survival = filtered["survived"].mean() * 100
    best_class = filtered.groupby("pclass")["survived"].mean().sort_values(ascending=False)
    best_gender = filtered.groupby("sex")["survived"].mean().sort_values(ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Passengers", f"{len(filtered):,}")
    c2.metric(
        "Survival Rate",
        f"{filtered_survival:.2f}%",
        delta=f"{filtered_survival - overall_survival:+.2f}% vs overall",
        delta_color="off",
    )
    c3.metric(
        "Female Survival",
        f"{filtered[filtered['sex'] == 'female']['survived'].mean() * 100:.2f}%" if (filtered["sex"] == "female").any() else "N/A",
    )
    c4.metric(
        "Male Survival",
        f"{filtered[filtered['sex'] == 'male']['survived'].mean() * 100:.2f}%" if (filtered["sex"] == "male").any() else "N/A",
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Exploration",
        "Model Lab",
        "Predict Passenger",
        "Field Guide",
    ])

    with tab1:
        st.subheader("Survival Breakdown")

        st.markdown(
            f"""
            <div class="pro-card">
                <div class="pro-title">Executive Summary</div>
                <div class="pro-text">
                    In the active filtered cohort, survival rate is <b>{filtered_survival:.2f}%</b>.
                    Best-performing class is <b>Pclass {int(best_class.index[0])}</b> at
                    <b>{best_class.iloc[0] * 100:.2f}%</b>, while top gender segment is
                    <b>{best_gender.index[0].title()}</b> at <b>{best_gender.iloc[0] * 100:.2f}%</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            survival_counts = filtered["survived"].value_counts().sort_index().reset_index()
            survival_counts.columns = ["survived", "count"]
            survival_counts["label"] = survival_counts["survived"].map({0: "Did Not Survive", 1: "Survived"})
            fig_bar = px.bar(
                survival_counts,
                x="label",
                y="count",
                color="survived",
                color_discrete_map=COLORS,
                title="Survival Count",
            )
            fig_bar.update_layout(showlegend=False, height=380)
            style_plot(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_pie = px.pie(
                survival_counts,
                names="label",
                values="count",
                color="label",
                color_discrete_map={"Did Not Survive": "#6f6f6f", "Survived": "#efefef"},
                hole=0.45,
                title="Survival Proportion",
            )
            fig_pie.update_traces(marker=dict(colors=["#6c6c6c", "#efefef"]))
            fig_pie.update_layout(height=380)
            style_plot(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Fare vs Age Bubble View")
        scatter = px.scatter(
            filtered,
            x="age",
            y="fare",
            size="family_size",
            color="survived",
            color_discrete_map=COLORS,
            hover_data=["sex", "pclass", "title"],
            title="Passenger Distribution: Age vs Fare",
        )
        scatter.update_layout(height=420)
        style_plot(scatter)
        st.plotly_chart(scatter, use_container_width=True)

        st.subheader("Top Rows (Filtered)")
        st.dataframe(filtered.head(20), use_container_width=True)

    with tab2:
        st.subheader("Exploratory Visuals")
        left, right = st.columns(2)

        with left:
            fig_gender = px.histogram(
                filtered,
                x="sex",
                color="survived",
                barmode="group",
                color_discrete_map=COLORS,
                title="Survival vs Gender",
            )
            fig_gender.update_layout(height=380)
            style_plot(fig_gender)
            st.plotly_chart(fig_gender, use_container_width=True)

            fig_age = px.histogram(
                filtered,
                x="age",
                nbins=28,
                color="survived",
                color_discrete_map=COLORS,
                title="Age Distribution by Survival",
            )
            fig_age.update_layout(height=380)
            style_plot(fig_age)
            st.plotly_chart(fig_age, use_container_width=True)

        with right:
            fig_class = px.histogram(
                filtered,
                x="pclass",
                color="survived",
                barmode="group",
                color_discrete_map=COLORS,
                title="Survival vs Passenger Class",
            )
            fig_class.update_layout(height=380)
            style_plot(fig_class)
            st.plotly_chart(fig_class, use_container_width=True)

            rates = (
                filtered.groupby("age_group", observed=False)["survived"].mean().reset_index().dropna()
            )
            rates["survival_pct"] = rates["survived"] * 100
            fig_rates = px.line(
                rates,
                x="age_group",
                y="survival_pct",
                markers=True,
                title="Survival Rate by Age Group (%)",
            )
            fig_rates.update_traces(line=dict(color="#efefef", width=3), marker=dict(size=9, color="#efefef"))
            fig_rates.update_layout(height=380, yaxis_title="Survival %", xaxis_title="Age Group")
            style_plot(fig_rates)
            st.plotly_chart(fig_rates, use_container_width=True)

        heatmap_data = filtered[["survived", "age", "fare", "family_size", "pclass", "sibsp", "parch"]].corr(numeric_only=True)
        corr_fig = px.imshow(
            heatmap_data,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Greys",
            title="Feature Correlation Heatmap",
        )
        corr_fig.update_layout(height=500)
        style_plot(corr_fig)
        st.plotly_chart(corr_fig, use_container_width=True)

    with tab3:
        st.subheader("Model Performance")
        perf_df = pd.DataFrame(
            {
                "Model": list(model_results.keys()),
                "Accuracy": [model_results[m]["accuracy"] for m in model_results],
                "AUC": [model_results[m]["auc"] for m in model_results],
            }
        ).sort_values("Accuracy", ascending=False)

        fig_perf = px.bar(
            perf_df,
            x="Model",
            y="Accuracy",
            hover_data=["AUC"],
            color="Model",
            text=perf_df["Accuracy"].map(lambda x: f"{x:.4f}"),
            title="Accuracy Comparison",
            color_discrete_sequence=["#efefef", "#7a7a7a"],
        )
        fig_perf.update_layout(height=380, showlegend=False)
        style_plot(fig_perf)
        st.plotly_chart(fig_perf, use_container_width=True)

        st.plotly_chart(make_roc_chart(model_results, _y_test), use_container_width=True)

        m1, m2 = st.columns(2)
        names = list(model_results.keys())
        with m1:
            st.plotly_chart(
                make_confusion_chart(
                    model_results[names[0]]["confusion_matrix"],
                    f"{names[0]} Confusion Matrix",
                ),
                use_container_width=True,
            )
        with m2:
            st.plotly_chart(
                make_confusion_chart(
                    model_results[names[1]]["confusion_matrix"],
                    f"{names[1]} Confusion Matrix",
                ),
                use_container_width=True,
            )

        selected_model = st.selectbox("Show detailed classification report", options=names)
        report_df = pd.DataFrame(model_results[selected_model]["report"]).T
        st.dataframe(report_df, use_container_width=True)

        if "Decision Tree" in model_results:
            dt_importance = pd.DataFrame(
                {
                    "Feature": features,
                    "Importance": model_results["Decision Tree"]["model"].feature_importances_,
                }
            ).sort_values("Importance", ascending=False)
            fig_imp = px.bar(
                dt_importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Decision Tree Feature Importance",
                color="Importance",
                color_continuous_scale="Greys",
            )
            fig_imp.update_layout(height=420)
            style_plot(fig_imp)
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab4:
        st.subheader("Passenger Survival Predictor")
        st.caption("Enter passenger details and compare model predictions with probability confidence")

        input_col, board_col = st.columns([1.1, 0.9], gap="large")

        with input_col:
            col1, col2, col3 = st.columns(3)
            with col1:
                pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
                sex_label = st.selectbox("Sex", ["female", "male"], index=1)
                age = st.slider("Age", min_value=1, max_value=80, value=30)
                fare = st.slider("Fare", min_value=0.0, max_value=600.0, value=32.0)
            with col2:
                sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
                parch = st.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
                embarked = st.selectbox("Embarked", ["C", "Q", "S"], index=2)
                has_cabin = st.selectbox("Has Cabin Info", [0, 1], index=0)
            with col3:
                title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"], index=0)
                age_group = st.selectbox("Age Group", ["Child", "Teen", "Young_Adult", "Middle_Age", "Senior"], index=2)
                family_size = sibsp + parch
                st.metric("Computed Family Size", family_size)

        input_df = pd.DataFrame(
            [
                {
                    "pclass": pclass,
                    "sex": 1 if sex_label == "female" else 0,
                    "age": age,
                    "sibsp": sibsp,
                    "parch": parch,
                    "fare": fare,
                    "embarked": {"C": 0, "Q": 1, "S": 2}[embarked],
                    "family_size": family_size,
                    "has_cabin": has_cabin,
                    "title": {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}[title],
                    "age_group": {
                        "Child": 0,
                        "Teen": 1,
                        "Young_Adult": 2,
                        "Middle_Age": 3,
                        "Senior": 4,
                    }[age_group],
                }
            ]
        )[features]

        prediction_payload = []
        for model_name in model_results:
            model = model_results[model_name]["model"]
            prob_survive = model.predict_proba(input_df)[0][1]
            pred_label = "Survive" if prob_survive >= 0.5 else "Not Survive"
            prediction_payload.append(
                {
                    "model": model_name,
                    "prob": prob_survive,
                    "label": pred_label,
                }
            )

        with board_col:
            st.markdown(
                """
                <div class="pro-card">
                    <div class="pro-title">Classification Board</div>
                    <div class="pro-text">Model decisions for the selected passenger profile.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            criteria_df = pd.DataFrame(
                {
                    "Criteria": [
                        "Passenger Class",
                        "Sex",
                        "Age",
                        "Fare",
                        "Embarked",
                        "Title",
                        "Family Size",
                        "Has Cabin Info",
                    ],
                    "Value": [
                        f"Pclass {pclass}",
                        sex_label.title(),
                        age,
                        f"{fare:.2f}",
                        embarked,
                        title,
                        family_size,
                        "Yes" if has_cabin == 1 else "No",
                    ],
                }
            )
            st.dataframe(criteria_df, use_container_width=True, hide_index=True)

            board_cols = st.columns(2)
            for idx, payload in enumerate(prediction_payload):
                bg = "#141414" if payload["label"] == "Not Survive" else "#1c1c1c"
                border = "#3a3a3a" if payload["label"] == "Not Survive" else "#5a5a5a"
                board_cols[idx].markdown(
                    f"""
                    <div style=\"background:{bg}; border:1px solid {border}; border-radius:12px; padding:14px; min-height:120px;\">
                        <div style=\"font-size:14px; color:#d9d9d9;\">{payload['model']}</div>
                        <div style=\"font-size:28px; font-weight:700; color:#f3f3f3; margin-top:4px;\">{payload['label']}</div>
                        <div style=\"font-size:13px; color:#bdbdbd; margin-top:6px;\">Survival Probability: {payload['prob'] * 100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            avg_prob = sum(x["prob"] for x in prediction_payload) / len(prediction_payload)
            consensus = "Survive" if avg_prob >= 0.5 else "Not Survive"
            st.markdown(
                f"""
                <div class=\"pro-card\" style=\"margin-top:12px;\">
                    <div class=\"pro-title\">Consensus Outcome</div>
                    <div class=\"pro-text\">{consensus} (average survival probability: {avg_prob * 100:.2f}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab5:
        st.subheader("Feature Field Guide")
        st.caption("Simple explanation of each input used in this dashboard and model.")

        st.markdown(
            """
            <div class="pro-card">
                <div class="pro-title">What is Pclass?</div>
                <div class="pro-text">
                    <b>Pclass</b> means passenger ticket class.
                    It is a proxy for ticket level and access on the ship.
                    <br><br>
                    <b>1</b> = First Class (highest tier)<br>
                    <b>2</b> = Second Class<br>
                    <b>3</b> = Third Class (lowest tier)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                """
                <div class="pro-card">
                    <div class="pro-title">Title Meanings (from passenger names)</div>
                    <div class="pro-text">
                        <b>Mr</b>: Adult male title<br>
                        <b>Mrs</b>: Married adult female title<br>
                        <b>Miss</b>: Unmarried female title<br>
                        <b>Master</b>: Young boy title<br>
                        <b>Rare</b>: Less common titles (Dr, Rev, Col, Lady, etc.)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="pro-card">
                    <div class="pro-title">Embarked (Port Code)</div>
                    <div class="pro-text">
                        <b>C</b>: Cherbourg<br>
                        <b>Q</b>: Queenstown<br>
                        <b>S</b>: Southampton
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                """
                <div class="pro-card">
                    <div class="pro-title">Family-related Fields</div>
                    <div class="pro-text">
                        <b>SibSp</b>: Number of siblings/spouses aboard<br>
                        <b>Parch</b>: Number of parents/children aboard<br>
                        <b>Family Size</b>: SibSp + Parch<br>
                        <b>Has Cabin Info</b>: 1 means cabin known, 0 means unknown
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="pro-card">
                    <div class="pro-title">Target Label</div>
                    <div class="pro-text">
                        <b>Survived = 1</b>: Passenger survived<br>
                        <b>Survived = 0</b>: Passenger did not survive
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        field_rows = [
            ("pclass", "Passenger ticket class (1, 2, 3)"),
            ("sex", "Passenger sex category"),
            ("age", "Passenger age in years"),
            ("fare", "Ticket fare paid"),
            ("title", "Title extracted from name"),
            ("embarked", "Boarding port code (C/Q/S)"),
            ("sibsp", "Siblings/spouses aboard"),
            ("parch", "Parents/children aboard"),
            ("family_size", "Derived family count onboard"),
            ("age_group", "Derived age band"),
            ("survived", "Model target variable"),
        ]
        glossary_df = pd.DataFrame(field_rows, columns=["Field", "Meaning"])
        st.dataframe(glossary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Built with Streamlit + Plotly | Dataset: Titanic - Machine Learning from Disaster")


if __name__ == "__main__":
    main()
