"""
I use this Streamlit app to tell the hotel no-show EDA story.

I turn the notebook findings into an interactive walkthrough that connects
data quality, imputation, feature engineering, visual insights, and model
evaluation decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "noshow.csv"

SECTIONS = [
    "Overview",
    "Introduction & Setup",
    "First look at the data",
    "Data cleaning and Fixing Errors",
    "Smart Data Imputation",
    "Feature Engineering",
    "Visual Analysis & Core Insights",
    "Conclusions and Pipeline Blueprint",
]

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

MONTH_TO_NUM = {month: index + 1 for index, month in enumerate(MONTH_ORDER)}


st.set_page_config(
    page_title="Hotel No-Show EDA Story",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """
    I load the hotel no-show CSV file.

    Returns:
        pd.DataFrame: Raw booking records.
    """
    return pd.read_csv(DATA_PATH)


def parse_price(value: object) -> tuple[object, float]:
    """
    I parse the raw price text into currency and numeric amount.

    Args:
        value (object): Raw price value such as 'SGD$ 492.98'.

    Returns:
        tuple[object, float]: Currency token and parsed numeric amount.
    """
    if pd.isna(value):
        return np.nan, np.nan

    parts = str(value).split()
    if len(parts) != 2:
        return np.nan, np.nan

    try:
        return parts[0], float(parts[1])
    except ValueError:
        return np.nan, np.nan


@st.cache_data(show_spinner=False)
def prepare_story_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    I apply lightweight EDA cleaning used for visualization.

    Returns:
        pd.DataFrame: Cleaned data with parsed price and engineered features.
    """
    story_df = df.copy()
    story_df = story_df.dropna(subset=["no_show"]).copy()

    for column in ["booking_month", "arrival_month", "checkout_month"]:
        story_df[column] = story_df[column].astype(str).str.lower().str.capitalize()

    story_df["arrival_day"] = story_df["arrival_day"].abs()
    story_df["checkout_day"] = story_df["checkout_day"].abs()
    story_df["num_adults"] = story_df["num_adults"].replace({"one": 1, "two": 2}).astype(int)

    price_parts = story_df["price"].apply(parse_price)
    story_df["currency"] = price_parts.apply(lambda item: item[0])
    story_df["price_value"] = price_parts.apply(lambda item: item[1])
    story_df["price_sgd"] = np.where(
        story_df["currency"].eq("USD$"),
        story_df["price_value"] * 1.3718,
        story_df["price_value"],
    )

    story_df["arrival_month_num"] = story_df["arrival_month"].map(MONTH_TO_NUM)
    story_df["booking_month_num"] = story_df["booking_month"].map(MONTH_TO_NUM)
    story_df["checkout_month_num"] = story_df["checkout_month"].map(MONTH_TO_NUM)
    story_df["lead_time_months"] = (
        story_df["arrival_month_num"] - story_df["booking_month_num"]
    ) % 12

    story_df["stay_duration"] = (
        (story_df["checkout_month_num"] - story_df["arrival_month_num"]) * 31
        + story_df["checkout_day"]
        - story_df["arrival_day"]
    )
    story_df.loc[story_df["stay_duration"] <= 0, "stay_duration"] = 1

    return story_df


def section_header(title: str, caption: str) -> None:
    """
    I render a consistent page title and short purpose statement.

    Args:
        title (str): Section title.
        caption (str): Section purpose.
    """
    st.title(title)
    st.caption(caption)


def insight_box(what: str, why: str, action: str) -> None:
    """
    I render the required what, why, and action interpretation block.

    Args:
        what (str): What the chart or metric shows.
        why (str): Why it matters.
        action (str): Supported action or insight.
    """
    st.info(
        f"**What it shows:** {what}\n\n"
        f"**Why it matters:** {why}\n\n"
        f"**Action or insight:** {action}"
    )


def rate_table(df: pd.DataFrame, group_col: str, top_n: int | None = None) -> pd.DataFrame:
    """
    I create a no-show rate table for a selected group.

    Args:
        df (pd.DataFrame): Story dataframe.
        group_col (str): Column used for grouping.
        top_n (int | None): Optional number of rows to return.

    Returns:
        pd.DataFrame: Count and no-show rate by group.
    """
    table = (
        df.groupby(group_col, dropna=False)
        .agg(bookings=("booking_id", "count"), no_show_rate=("no_show", "mean"))
        .sort_values("no_show_rate", ascending=False)
    )
    table["no_show_rate"] = table["no_show_rate"] * 100
    if top_n:
        table = table.head(top_n)
    return table


raw_df = load_raw_data()
df = prepare_story_data(raw_df)

st.sidebar.title("Hotel No-Show EDA")
selected_section = st.sidebar.radio("Story sections", SECTIONS, index=0)
st.sidebar.divider()
st.sidebar.metric("Raw rows", f"{len(raw_df):,}")
st.sidebar.metric("Clean EDA rows", f"{len(df):,}")
st.sidebar.metric("No-show rate", f"{df['no_show'].mean() * 100:.1f}%")


if selected_section == "Overview":
    section_header(
        "Hotel No-Show Prediction: EDA Story",
        "A visual walkthrough of the key data findings, model evaluation results, and pipeline decisions from eda.ipynb.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bookings loaded", f"{len(raw_df):,}")
    col2.metric("Columns", f"{raw_df.shape[1]}")
    col3.metric("Overall no-show rate", f"{df['no_show'].mean() * 100:.1f}%")
    col4.metric("Missing price rows", f"{df['price_sgd'].isna().sum():,}")

    st.subheader("Main story")
    st.write(
        "I use the EDA to explain why hotel no-shows are not random. The data shows "
        "clear patterns by branch, customer origin, booking platform, booking lead time, "
        "room type, and price. I then connect these findings to cleaning choices, smart "
        "imputation, feature engineering, and model selection."
    )

    target_counts = df["no_show"].map({0.0: "Showed up", 1.0: "No-show"}).value_counts()
    st.bar_chart(target_counts)
    insight_box(
        "About 37% of bookings are no-shows while about 63% show up.",
        "The target is imbalanced, so accuracy alone can be misleading.",
        "I compare models using ROC-AUC and F1 in addition to accuracy, because the hotel needs useful risk ranking and no-show detection.",
    )

    st.subheader("Evaluation summary")
    eval_df = pd.DataFrame(
        {
            "Model": ["LightGBM Classifier", "Random Forest Classifier", "Logistic Regression"],
            "F1-Score": [0.6033, 0.6041, 0.5880],
            "ROC-AUC": [0.7697, 0.7689, 0.7474],
            "Accuracy": [0.7410, 0.7401, 0.7271],
            "Decision": ["Selected", "Candidate", "Candidate"],
        }
    )
    st.dataframe(eval_df, hide_index=True, use_container_width=True)
    insight_box(
        "LightGBM has the strongest ROC-AUC and accuracy, while Random Forest has a slightly higher F1-score by 0.0008.",
        "ROC-AUC is the main metric because it measures how well I can rank bookings by no-show risk.",
        "I select LightGBM because it gives strong ranking performance and is lighter to deploy than a larger forest model.",
    )


elif selected_section == "Introduction & Setup":
    section_header(
        "Introduction & Setup",
        "This section frames the business problem and explains how the raw booking data is loaded.",
    )

    st.write(
        "The business problem is simple: when guests do not arrive, the hotel loses room "
        "revenue and cannot plan capacity confidently. My EDA objective is to find patterns "
        "that help predict which bookings are more likely to become no-shows."
    )

    st.subheader("Data source")
    st.write(
        "The notebook connects to `data/noshow.db`; this app reads the matching "
        "`data/noshow.csv` so the story can run directly in Streamlit."
    )
    st.code("data/noshow.csv", language="text")

    st.subheader("EDA workflow")
    workflow = pd.DataFrame(
        {
            "Stage": [
                "Load data",
                "Inspect schema",
                "Fix quality issues",
                "Impute missing values",
                "Create features",
                "Visualize patterns",
                "Define model pipeline",
            ],
            "Purpose": [
                "Bring raw bookings into a dataframe.",
                "Understand columns, missingness, and data types.",
                "Standardize months, days, guest counts, and prices.",
                "Keep useful rows instead of dropping large missing groups.",
                "Capture stay length and booking lead time.",
                "Identify the clearest no-show drivers.",
                "Turn EDA findings into a reproducible ML pipeline.",
            ],
        }
    )
    st.dataframe(workflow, hide_index=True, use_container_width=True)


elif selected_section == "First look at the data":
    section_header(
        "First Look at the Data",
        "This section checks shape, sample records, schema, missing values, and target balance.",
    )

    st.subheader("Dataset preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Column types")
        schema = raw_df.dtypes.astype(str).rename("dtype").to_frame()
        st.dataframe(schema, use_container_width=True)
    with col2:
        st.subheader("Missing values")
        missing = raw_df.isna().sum().sort_values(ascending=False)
        st.bar_chart(missing)

    insight_box(
        "The dataset has one almost empty row, 21,612 missing room values, and 24,881 missing price values.",
        "Dropping all rows with missing room or price would remove a large amount of useful booking behavior.",
        "I drop only the empty target row, then treat room and price imputation as a modeling problem.",
    )

    st.subheader("Target distribution")
    target_distribution = (
        df["no_show"].map({0.0: "Showed up", 1.0: "No-show"}).value_counts(normalize=True)
        * 100
    )
    st.bar_chart(target_distribution)
    insight_box(
        "The target split is roughly 63% showed up and 37% no-show.",
        "This is not extremely rare, but it is still imbalanced enough that a naive model can look better than it is.",
        "I keep a baseline-first evaluation mindset and avoid judging performance from accuracy alone.",
    )


elif selected_section == "Data cleaning and Fixing Errors":
    section_header(
        "Data Cleaning and Fixing Errors",
        "This section explains the practical fixes needed before modeling.",
    )

    cleaning_steps = pd.DataFrame(
        {
            "Issue": [
                "One empty target row",
                "Mixed month capitalization",
                "Negative checkout days",
                "Text values in num_adults",
                "Mixed SGD and USD price strings",
            ],
            "Fix": [
                "Drop the row because it has no usable target.",
                "Convert month text to title case.",
                "Use absolute values because calendar days should be positive.",
                "Map 'one' and 'two' to 1 and 2.",
                "Parse currency and convert USD to SGD using the notebook rate.",
            ],
            "Why it matters": [
                "The model needs a valid label to learn from the row.",
                "Consistent categories prevent duplicate month labels.",
                "Invalid dates distort stay duration calculations.",
                "Numeric guest counts are needed for imputation and modeling.",
                "Prices must be comparable before they can support room and price analysis.",
            ],
        }
    )
    st.dataframe(cleaning_steps, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Arrival month before cleaning")
        st.bar_chart(raw_df["arrival_month"].value_counts().head(15))
    with col2:
        st.subheader("Arrival month after cleaning")
        st.bar_chart(df["arrival_month"].value_counts().reindex(MONTH_ORDER).dropna())

    insight_box(
        "Variants such as 'MaY', 'JunE', and 'JulY' collapse into standard month names after cleaning.",
        "Without this step, the model would treat spelling and capitalization errors as separate categories.",
        "I standardize text categories before feature engineering and encoding.",
    )

    currency_counts = df["currency"].fillna("Missing").value_counts()
    st.subheader("Currency frequency before price unification")
    st.bar_chart(currency_counts)
    insight_box(
        "Known prices are split almost evenly between SGD and USD, with 24,881 missing prices.",
        "A numeric price column is only meaningful when all values share the same currency scale.",
        "I convert USD to SGD before using price for imputation, analysis, and modeling.",
    )


elif selected_section == "Smart Data Imputation":
    section_header(
        "Smart Data Imputation",
        "This section explains why the notebook uses model-based imputation for room and price.",
    )

    both_missing = raw_df[raw_df["room"].isna() & raw_df["price"].isna()]
    room_missing_price_known = raw_df[raw_df["room"].isna() & raw_df["price"].notna()]
    price_missing_room_known = raw_df[raw_df["price"].isna() & raw_df["room"].notna()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Both room and price missing", f"{len(both_missing):,}")
    col2.metric("Room missing, price known", f"{len(room_missing_price_known):,}")
    col3.metric("Price missing, room known", f"{len(price_missing_room_known):,}")

    st.subheader("Why this missingness pattern is useful")
    st.write(
        "The missing values are complementary: room is missing when price is usually known, "
        "and price is missing when room is usually known. This means each field can help "
        "recover the other instead of forcing a simple mean or mode fill."
    )

    imputation_plan = pd.DataFrame(
        {
            "Missing field": ["room", "price_sgd"],
            "Notebook method": ["KNN Classifier", "Random Forest Regressor"],
            "Inputs used": [
                "Price and hotel branch",
                "Room type, branch, number of adults, and number of children",
            ],
            "Trade-off": [
                "More realistic than mode fill, but depends on the quality of nearby price and branch patterns.",
                "More flexible than median fill, but more complex and should be validated against a simple baseline.",
            ],
        }
    )
    st.dataframe(imputation_plan, hide_index=True, use_container_width=True)

    st.subheader("Room breakdown after notebook imputation")
    imputed_room_counts = pd.Series(
        {
            "King": 85998,
            "Single": 19202,
            "Queen": 13259,
            "President Suite": 931,
        }
    )
    st.bar_chart(imputed_room_counts)
    insight_box(
        "After imputation, room and price have no remaining null values in the notebook workflow.",
        "This preserves more training data than dropping over 20% of rows.",
        "I use model-based imputation as an experiment, and I would compare it against simple mode and median baselines before finalizing it.",
    )


elif selected_section == "Feature Engineering":
    section_header(
        "Feature Engineering",
        "This section shows how raw booking dates are converted into more useful behavior signals.",
    )

    feature_plan = pd.DataFrame(
        {
            "Feature": ["stay_duration", "lead_time_months"],
            "Built from": [
                "arrival_month, arrival_day, checkout_month, checkout_day",
                "booking_month and arrival_month",
            ],
            "Why I use it": [
                "Stay length can reflect guest commitment and booking type.",
                "Booking lead time is a common no-show signal because plans made far ahead can change.",
            ],
            "Trade-off": [
                "Month boundary logic must be handled carefully to avoid invalid durations.",
                "Month-level lead time is simple and explainable, but it is less precise than full date differences.",
            ],
        }
    )
    st.dataframe(feature_plan, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stay duration distribution")
        st.bar_chart(df["stay_duration"].clip(upper=14).value_counts().sort_index())
    with col2:
        st.subheader("Lead time distribution")
        st.bar_chart(df["lead_time_months"].value_counts().sort_index())

    insight_box(
        "Most stays are short, with a median around 2 days, while lead time spans 0 to 11 months.",
        "Short stays and long lead times may represent different guest behavior patterns.",
        "I keep both features because they are simple, interpretable, and directly connected to the booking journey.",
    )


elif selected_section == "Visual Analysis & Core Insights":
    section_header(
        "Visual Analysis & Core Insights",
        "This section highlights the main no-show drivers found in the notebook.",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Branch and platform", "Country", "Lead time", "Price and room"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("No-show rate by branch")
            branch_rate = rate_table(df, "branch")
            st.bar_chart(branch_rate["no_show_rate"])
        with col2:
            st.subheader("No-show rate by platform")
            platform_rate = rate_table(df, "platform")
            st.bar_chart(platform_rate["no_show_rate"])

        insight_box(
            "Changi has a higher no-show rate than Orchard, and platform behavior differs across channels.",
            "Location and booking channel can represent different customer segments and commitment levels.",
            "I include branch and platform-style behavior in the analysis and consider branch as a strong candidate feature.",
        )

    with tab2:
        st.subheader("No-show rate by country")
        country_rate = rate_table(df, "country")
        st.bar_chart(country_rate["no_show_rate"])
        st.dataframe(country_rate, use_container_width=True)
        insight_box(
            "The notebook highlights China at about 56.6% no-show and Japan at about 17.1%.",
            "Customer origin is one of the clearest segmentation signals in the EDA.",
            "I encode country for modeling, while treating it as a risk signal rather than a causal explanation.",
        )

    with tab3:
        st.subheader("No-show rate by booking lead time")
        lead_rate = rate_table(df, "lead_time_months").sort_index()
        st.line_chart(lead_rate["no_show_rate"])
        insight_box(
            "Same-month bookings have the lowest no-show risk, while bookings made many months ahead are riskier.",
            "Longer lead time gives guests more time for plans to change.",
            "I engineer lead_time_months as a simple behavioral feature and monitor whether it improves validation metrics.",
        )

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Median price by room")
            st.bar_chart(df.groupby("room")["price_sgd"].median().sort_values())
        with col2:
            st.subheader("Median price by branch and room")
            price_pivot = df.pivot_table(
                index="room",
                columns="branch",
                values="price_sgd",
                aggfunc="median",
            )
            st.dataframe(price_pivot.style.format("{:.2f}"), use_container_width=True)

        insight_box(
            "President Suite is the highest-priced room type, followed by King, Queen, and Single; Orchard prices are generally higher than Changi.",
            "This structure explains why room and branch are useful for estimating missing prices.",
            "I use room, branch, and guest counts to support price imputation, then scale price for model training.",
        )


else:
    section_header(
        "Conclusions and Pipeline Blueprint",
        "This section converts the EDA findings into a practical modeling workflow.",
    )

    st.subheader("Key conclusions")
    conclusions = pd.DataFrame(
        {
            "Finding": [
                "No-shows are common enough to matter operationally.",
                "Branch, country, platform, and lead time show meaningful risk differences.",
                "Room and price missingness should not be handled by dropping rows.",
                "Currency and date cleanup are required before modeling.",
                "LightGBM is the selected model from the recorded evaluation.",
            ],
            "Pipeline decision": [
                "Use classification with ranking-focused metrics.",
                "Encode categorical booking signals and engineer lead time.",
                "Use smart imputation and compare it with simple baselines.",
                "Standardize categories, prices, and date-derived features.",
                "Use ROC-AUC as the primary selection metric, while monitoring F1 and accuracy.",
            ],
        }
    )
    st.dataframe(conclusions, hide_index=True, use_container_width=True)

    st.subheader("Pipeline blueprint")
    st.markdown(
        """
```mermaid
flowchart TD
A[Raw booking data] --> B[Ingestion]
B --> C[Cleaning]
C --> D[Smart imputation]
D --> E[Feature engineering]
E --> F[Train-test split]
F --> G[Preprocessing]
G --> H[Baseline model]
H --> I[Candidate models]
I --> J[Evaluation]
J --> K[Best model selection]
K --> L[Inference artifacts]
```
"""
    )

    st.subheader("Model evaluation interpretation")
    confusion = pd.DataFrame(
        {
            "Predicted show up": [12991, 4142],
            "Predicted no-show": [2042, 4703],
        },
        index=["Actual show up", "Actual no-show"],
    )
    st.dataframe(confusion, use_container_width=True)
    insight_box(
        "The selected model correctly identifies 4,703 no-shows, but it also misses 4,142 no-shows and raises 2,042 false alarms.",
        "False negatives mean empty rooms and lost revenue; false positives can create operational risk if overbooking is too aggressive.",
        "I would tune the decision threshold based on the hotel's cost trade-off between empty rooms and false alarms.",
    )

    st.subheader("Recommended next experiments")
    st.write(
        "I would validate simple median and mode imputation against KNN and Random Forest imputation, "
        "compare each candidate model against a baseline, and tune the classification threshold using "
        "business costs. This keeps the pipeline evidence-driven instead of assuming a complex method is always better."
    )
