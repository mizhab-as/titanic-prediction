import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


SURVIVE_COLORS = ["#D1495B", "#2A9D8F"]


def print_section(title: str) -> None:
    border = "=" * 78
    print(f"\n{border}")
    print(title.center(78))
    print(border)


def format_rate_table(series: pd.Series, name: str) -> str:
    table = (series * 100).round(2).reset_index()
    table.columns = [name, "survival_rate_pct"]
    return table.to_string(index=False)


def get_output_dir() -> Path:
    output_dir = Path(__file__).resolve().parents[1] / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    fallback_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded local dataset: {data_path}")
    else:
        print("Local dataset not found. Downloading fallback Titanic train dataset...")
        df = pd.read_csv(fallback_url)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Saved dataset to: {data_path}")

    df.columns = df.columns.str.lower()
    return df


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    return match.group(1).strip() if match else "Unknown"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["age"] = data["age"].fillna(data["age"].median())
    data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
    data["cabin"] = data["cabin"].fillna("Unknown")

    data["title"] = data["name"].apply(extract_title)
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    data["title"] = data["title"].replace(title_map)
    data["title"] = data["title"].where(
        data["title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare"
    )

    data["family_size"] = data["sibsp"] + data["parch"]
    data["has_cabin"] = (data["cabin"] != "Unknown").astype(int)
    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young_Adult", "Middle_Age", "Senior"],
    )

    drop_cols = [c for c in ["ticket", "cabin", "passengerid", "name"] if c in data.columns]
    data = data.drop(columns=drop_cols)
    return data


def run_eda(data: pd.DataFrame) -> None:
    print_section("EXPLORATORY DATA ANALYSIS")

    gender_rates = data.groupby("sex")["survived"].mean().sort_values(ascending=False)
    class_rates = data.groupby("pclass")["survived"].mean().sort_values(ascending=False)
    age_rates = data.groupby("age_group", observed=False)["survived"].mean()

    print("\nSurvival by gender (%):")
    print(format_rate_table(gender_rates, "sex"))

    print("\nSurvival by passenger class (%):")
    print(format_rate_table(class_rates, "pclass"))

    print("\nSurvival by age group (%):")
    print(format_rate_table(age_rates, "age_group"))


def plot_visuals(data: pd.DataFrame, output_dir: Path) -> None:
    print_section("VISUALIZATION EXPORT")
    sns.set_theme(style="whitegrid", context="talk")

    # Dashboard style figure for quick storytelling.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Titanic Survival Snapshot", fontsize=22, fontweight="bold")

    counts = data["survived"].value_counts().sort_index()
    axes[0, 0].bar(["Did Not Survive", "Survived"], counts.values, color=SURVIVE_COLORS)
    axes[0, 0].set_title("Survival Counts")
    axes[0, 0].set_ylabel("Passengers")
    for idx, value in enumerate(counts.values):
        axes[0, 0].text(idx, value + 8, str(value), ha="center", fontsize=11)

    sns.countplot(data=data, x="sex", hue="survived", ax=axes[0, 1], palette=SURVIVE_COLORS)
    axes[0, 1].set_title("Survival vs Gender")
    axes[0, 1].legend(["Did Not Survive", "Survived"], title="Outcome")

    sns.countplot(data=data, x="pclass", hue="survived", ax=axes[1, 0], palette=SURVIVE_COLORS)
    axes[1, 0].set_title("Survival vs Passenger Class")
    axes[1, 0].set_xlabel("Passenger Class")
    axes[1, 0].legend(["Did Not Survive", "Survived"], title="Outcome")

    sns.histplot(
        data=data,
        x="age",
        bins=28,
        hue="survived",
        multiple="stack",
        palette=SURVIVE_COLORS,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Age Distribution by Survival")
    axes[1, 1].set_xlabel("Age")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_dir / "survival_dashboard.png", dpi=180)
    plt.close(fig)

    # Keep original exports too for compatibility.
    plt.figure(figsize=(6, 4))
    data["survived"].value_counts().sort_index().plot(kind="bar", color=SURVIVE_COLORS)
    plt.xticks([0, 1], ["Did Not Survive", "Survived"], rotation=0)
    plt.title("Survival Counts")
    plt.tight_layout()
    plt.savefig(output_dir / "survival_counts.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=data, x="sex", hue="survived", ax=axes[0], palette=SURVIVE_COLORS)
    axes[0].set_title("Survival vs Gender")
    axes[0].legend(["Did Not Survive", "Survived"], title="Outcome")
    sns.countplot(data=data, x="pclass", hue="survived", ax=axes[1], palette=SURVIVE_COLORS)
    axes[1].set_title("Survival vs Class")
    axes[1].legend(["Did Not Survive", "Survived"], title="Outcome")
    plt.tight_layout()
    plt.savefig(output_dir / "survival_vs_gender_class.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sns.histplot(data=data, x="age", bins=30, hue="survived", multiple="stack", palette=SURVIVE_COLORS)
    plt.title("Age Distribution by Survival")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png", dpi=160)
    plt.close()

    print(f"Saved enhanced charts to: {output_dir}")


def hidden_patterns(data: pd.DataFrame) -> None:
    children = data[data["age"] <= 12]
    women = data[(data["sex"] == "female") & (data["age"] > 12)]
    men = data[(data["sex"] == "male") & (data["age"] > 12)]

    print_section("HIDDEN PATTERN DISCOVERY")
    print("\nWomen and children survival pattern:")
    print(f"Children survival rate: {children['survived'].mean() * 100:.2f}%")
    print(f"Women survival rate: {women['survived'].mean() * 100:.2f}%")
    print(f"Adult men survival rate: {men['survived'].mean() * 100:.2f}%")

    class_rates = data.groupby("pclass")["survived"].mean().sort_index()
    print("\nClass survival rates:")
    print((class_rates * 100).round(2).to_string())
    print("Higher class helps, but does not guarantee survival.")


def train_models(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    model_df = data.copy()
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

    lr = LogisticRegression(max_iter=1000, random_state=42)
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)

    model_rows = []

    print_section("MODEL TRAINING AND EVALUATION")

    for name, model in [("Logistic Regression", lr), ("Decision Tree", dt)]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        model_rows.append({"model": name, "accuracy": round(acc, 4)})

        print(f"\n{name}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, pred, digits=4))

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            xticklabels=["Pred: 0", "Pred: 1"],
            yticklabels=["True: 0", "True: 1"],
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        chart_name = name.lower().replace(" ", "_") + "_confusion_matrix.png"
        plt.savefig(output_dir / chart_name, dpi=170)
        plt.close()

    model_comparison = pd.DataFrame(model_rows).sort_values("accuracy", ascending=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=model_comparison,
        x="model",
        y="accuracy",
        hue="model",
        palette=["#2A9D8F", "#E76F51"],
        legend=False,
    )
    plt.ylim(0.70, 0.90)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("")
    plt.ylabel("Accuracy")
    for idx, row in model_comparison.reset_index(drop=True).iterrows():
        plt.text(idx, row["accuracy"] + 0.004, f"{row['accuracy']:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_dir / "model_accuracy_comparison.png", dpi=170)
    plt.close()

    print("\nModel comparison:")
    print(model_comparison.to_string(index=False))
    return model_comparison


def main() -> None:
    print_section("TITANIC SURVIVAL PREDICTION PIPELINE")
    df = load_data()
    output_dir = get_output_dir()
    processed = preprocess(df)

    print_section("DATA PREPROCESSING")
    missing = processed.isna().sum()[processed.isna().sum() > 0]
    if missing.empty:
        print("No missing values remain after preprocessing.")
    else:
        print("Remaining missing values:")
        print(missing.to_string())

    run_eda(processed)
    plot_visuals(processed, output_dir)
    hidden_patterns(processed)
    model_results = train_models(processed, output_dir)

    print_section("FINAL SUMMARY")
    print(f"Processed rows: {len(processed)}")
    print(f"Best model: {model_results.iloc[0]['model']} ({model_results.iloc[0]['accuracy']:.4f})")
    print(f"All plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
