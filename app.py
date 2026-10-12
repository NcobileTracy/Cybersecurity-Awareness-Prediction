
import json
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime
import uuid


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cybersecurity Awareness Assessment",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================
MODEL_PATH = str(Path(__file__).parent / "cybersecurity_awareness_FINAL_FROZEN.joblib")
OPTIONS_PATH = Path(__file__).parent / "dropdown_options.json"


# FINAL MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "1)   What is your Gender?",
    "2)    What is your Age?",
    "3)    Education Level ",
    "4)   Field of Study",
    "5)  University Name",
    "6)  How often do you use the Internet and Internet-related services? (E.g. Email, WhatsApp, News, YouTube  ",
    "7) What are your Internet/ Digital devices skills level?",
    "13) How secure do you feel your digital devices (e.g. computers and phones) are? ",
    "15)  What digital devices do you have Internet security on (e.g. anti-virus)?",
    "16)   If you use Internet security (e.g. anti-virus), is this kept up to date in terms of threat filters and signatures?",
    "21)  What do you feel about the threat of cyber crimes in the future?"
]


# DISPLAY LABELS
# ============================================================

DISPLAY_LABELS = {
    "1)   What is your Gender?":
        "What is your gender?",

    "2)    What is your Age?":
        "What is your age?",

    "3)    Education Level ":
        "What is your education level?",

    "4)   Field of Study":
        "What is your field of study?",

    "5)  University Name":
        "What is your university name?",

    "6)  How often do you use the Internet and Internet-related services? (E.g. Email, WhatsApp, News, YouTube  ":
        "How often do you use the Internet and Internet-related services?",

    "7) What are your Internet/ Digital devices skills level?":
        "What is your Internet / digital device skills level?",

    "13) How secure do you feel your digital devices (e.g. computers and phones) are? ":
        "How secure do you feel your digital devices are?",

    "15)  What digital devices do you have Internet security on (e.g. anti-virus)?":
        "Which digital devices have Internet security installed?",

    "16)   If you use Internet security (e.g. anti-virus), is this kept up to date in terms of threat filters and signatures?":
        "Is your Internet security kept up to date?",

    "21)  What do you feel about the threat of cyber crimes in the future?":
        "What do you think about the threat of cybercrime in the future?"
}


# ============================================================
# QUESTION GROUPS
# ============================================================

QUESTION_GROUPS = {
    "Personal Information": EXPECTED_FEATURES[:5],

    "Internet & Digital Skills": EXPECTED_FEATURES[5:7],

    "Digital Security": EXPECTED_FEATURES[7:]
}


# ============================================================
# MODEL METRICS
# ============================================================

TEST_ACCURACY = 0.6803

CV_MEAN = 0.6914

CV_STD = 0.0270

CV_SCORES = [
    0.69672131,
    0.67213115,
    0.69672131,
    0.65573770,
    0.73553719
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f6f8;
        color: #111111;
    }

    section[data-testid="stSidebar"] {
        background-color: #111111;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #111111;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #555555;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #111111;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #c00000;
    }

    .metric-label {
        color: #555555;
        font-size: 14px;
    }

    .result-card {
        background: #ffffff;
        padding: 30px;
        border-radius: 16px;
        border-left: 7px solid #c00000;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .result-level {
        font-size: 34px;
        font-weight: 800;
        color: #c00000;
    }

    .info-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        margin-bottom: 18px;
    }

    .small-muted {
        color: #666666;
        font-size: 14px;
    }

    div.stButton > button {
        background-color: #c00000;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 700;
        padding: 10px 20px;
    }

    div.stButton > button:hover {
        background-color: #8f0000;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "records" not in st.session_state:
    st.session_state.records = []

if "reviews" not in st.session_state:
    st.session_state.reviews = []


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)

    # Support a packaged model or a directly saved classifier
    if isinstance(package, dict):
        if "model" not in package:
            raise ValueError(
                "The saved package does not contain a 'model' entry."
            )
        return package["model"]

    return package


try:
    model = load_model()
except Exception as e:
    st.error("The prediction model could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# LOAD DROPDOWN OPTIONS
# ============================================================

try:
    with open(OPTIONS_PATH, "r", encoding="utf-8") as options_file:
        dropdown_options = json.load(options_file)
except Exception as e:
    st.error("The assessment options could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# GET DATASET OPTIONS
# ============================================================

def get_options(column):
    return dropdown_options.get(column, [])


# ============================================================


# MODEL CATEGORY FALLBACK
# ============================================================

def get_prediction_category(prediction):
    prediction = str(prediction).strip().lower()

    if "low" in prediction:
        return "Low"

    if "high" in prediction:
        return "High"

    return "Medium"


# ============================================================
# STRATEGIES
# ============================================================

def get_strategy(level):

    if level == "Low":
        return [
            "Use strong and unique passwords for every account.",
            "Enable multi-factor authentication whenever possible.",
            "Keep operating systems, applications and security software updated.",
            "Be careful when opening links or attachments from unknown sources.",
            "Avoid entering sensitive information on suspicious websites."
        ]

    if level == "Medium":
        return [
            "Continue using strong and unique passwords.",
            "Keep antivirus and security tools updated.",
            "Review suspicious emails, messages and links before opening them.",
            "Use secure networks when accessing sensitive information.",
            "Regularly review your privacy and security settings."
        ]

    return [
        "Continue practising strong cybersecurity habits.",
        "Keep security software and devices updated.",
        "Remain alert to phishing and social-engineering attacks.",
        "Regularly review account security and privacy settings.",
        "Share cybersecurity awareness with others."
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:25px;font-weight:800;margin-bottom:25px;">
        🔒 Cybersecurity Awareness
        </div>
        """,
        unsafe_allow_html=True
    )

    navigation = [
        "🏠 Home",
        "📝 Awareness Assessment",
        "🤖 Prediction Results",
        "🧠 AI Insights",
        "🗂️ Assessment Records",
        "📈 Analytics Dashboard",
        "📊 Model Dashboard",
        "🛡️ Cybersecurity Guidance",
        "⭐ Feedback",
        "ℹ️ About & Methodology",
    ]

    # Make the active page visibly different from other pages.
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .nav-active {
            background: #1d4ed8;
            color: white;
            border: 1px solid #60a5fa;
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }
        section[data-testid="stSidebar"] .nav-inactive {
            background: transparent;
            color: inherit;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    for item in navigation:
        page_name = item.split(" ", 1)[1]
        is_active = st.session_state.page == page_name

        if is_active:
            st.markdown(
                f'<div class="nav-active">{item}  ✓</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="nav-inactive">{item}</div>',
                unsafe_allow_html=True
            )
            if st.button(
                "Open " + page_name,
                key="nav_" + page_name,
                use_container_width=True
            ):
                st.session_state.page = page_name
                st.rerun()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        '<div class="main-title">Cybersecurity Awareness Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Assess cybersecurity awareness using a machine-learning-based assessment platform.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
        <h3>Welcome</h3>
        <p>
        This platform provides an assessment of cybersecurity awareness
        based on selected digital behaviour, security practices and
        cybersecurity perceptions.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">11</div>
                <div class="metric-label">Assessment Predictors</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">3</div>
                <div class="metric-label">Awareness Levels</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">68.03%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    if st.button("Start Awareness Assessment", use_container_width=True):
        st.session_state.page = "Awareness Assessment"
        st.rerun()


# ============================================================
# AWARENESS ASSESSMENT
# ============================================================

elif st.session_state.page == "Awareness Assessment":

    st.markdown(
        '<div class="main-title">Awareness Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Please answer all questions before generating your assessment result.'
        '</div>',
        unsafe_allow_html=True
    )

    completed_questions = 0

    answers = {}

    for group_name, columns in QUESTION_GROUPS.items():

        st.markdown(
            f'<div class="section-title">{group_name}</div>',
            unsafe_allow_html=True
        )

        for column in columns:

            label = DISPLAY_LABELS.get(column, column)

            options = get_options(column)

            display_options = options.copy()

            # Education display only
            if column == EXPECTED_FEATURES[2]:

                display_options = [
                    "Undergraduate" if value == "UG"
                    else "Postgraduate" if value == "PG"
                    else value
                    for value in options
                ]

            # Restore the saved answer when editing an assessment.
            answer_key = "answer_" + str(EXPECTED_FEATURES.index(column))
            saved_answer = st.session_state.answers.get(column)

            if column == EXPECTED_FEATURES[2]:
                if saved_answer == "UG":
                    saved_display = "Undergraduate"
                elif saved_answer == "PG":
                    saved_display = "Postgraduate"
                else:
                    saved_display = saved_answer
            else:
                saved_display = saved_answer

            select_options = ["Select an option"] + display_options

            if saved_display in display_options:
                default_index = select_options.index(saved_display)
            else:
                default_index = 0

            selected_display = st.selectbox(
                label,
                select_options,
                index=default_index,
                key=answer_key
            )

            if selected_display != "Select an option":

                if column == EXPECTED_FEATURES[2]:

                    if selected_display == "Undergraduate":
                        selected = "UG"

                    elif selected_display == "Postgraduate":
                        selected = "PG"

                    else:
                        selected = selected_display

                else:
                    selected = selected_display

                answers[column] = selected

                completed_questions += 1

    st.progress(
        completed_questions / len(EXPECTED_FEATURES)
    )

    st.caption(
        f"{completed_questions} of {len(EXPECTED_FEATURES)} questions completed"
    )

    if st.button(
        "Predict Awareness",
        use_container_width=True
    ):

        missing = [
            column
            for column in EXPECTED_FEATURES
            if column not in answers
        ]

        if missing:

            st.error(
                "Please answer all questions before generating your prediction."
            )

        else:

            X_input = pd.DataFrame(
                [answers],
                columns=EXPECTED_FEATURES
            )

            try:

                prediction = model.predict(X_input)[0]

                probabilities_array = model.predict_proba(X_input)[0]

                classes = list(model.classes_)

                probability_df = pd.DataFrame({
                    "Awareness Level": classes,
                    "Probability (%)": probabilities_array * 100
                })

                probability_df = probability_df.sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.session_state.answers = answers
                st.session_state.prediction = str(prediction)
                st.session_state.probabilities = probability_df

                record = {
                    "ID": str(uuid.uuid4())[:8].upper(),
                    "Date": datetime.now().strftime(
                        "%d %B %Y • %H:%M"
                    ),
                    "Prediction": str(prediction),
                    "Confidence": round(
                        float(max(probabilities_array)) * 100,
                        2
                    )
                }

                st.session_state.records.append(record)

                st.session_state.page = "Prediction Results"

                st.rerun()

            except Exception as e:

                st.error(
                    "The prediction could not be generated."
                )

                st.exception(e)


# ============================================================
# PREDICTION RESULTS
# ============================================================

elif st.session_state.page == "Prediction Results":

    st.markdown(
        '<div class="main-title">Prediction Results</div>',
        unsafe_allow_html=True
    )

    prediction = st.session_state.prediction

    probability_df = st.session_state.probabilities

    if prediction is None or probability_df is None:

        st.info(
            "Complete the Awareness Assessment first."
        )

    else:

        level = get_prediction_category(prediction)

        confidence = float(
            probability_df["Probability (%)"].max()
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-muted">
                    Predicted Cybersecurity Awareness
                </div>
                <div class="result-level">
                    {level}
                </div>
                <div class="small-muted">
                    Prediction confidence: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Awareness Probability")

        bar_fig = px.bar(
            probability_df,
            x="Awareness Level",
            y="Probability (%)",
            text="Probability (%)"
        )

        bar_fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        bar_fig.update_layout(
            template="plotly_white",
            yaxis_title="Probability (%)",
            xaxis_title="Awareness Level",
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        # ====================================================
        # PIE CHART
        # ====================================================

        st.subheader("Awareness Probability Distribution")

        pie_fig = px.pie(
            probability_df,
            names="Awareness Level",
            values="Probability (%)",
            hole=0.35
        )

        pie_fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text="Awareness Level"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Interpretation")

        if level == "Low":

            st.write(
                "The assessment indicates a lower level of "
                "cybersecurity awareness. Additional attention "
                "to security practices is recommended."
            )

        elif level == "Medium":

            st.write(
                "The assessment indicates a moderate level of "
                "cybersecurity awareness. There are positive "
                "security practices, but some areas can still be improved."
            )

        else:

            st.write(
                "The assessment indicates a higher level of "
                "cybersecurity awareness and generally positive "
                "cybersecurity practices."
            )

        strategies = get_strategy(level)

        st.subheader("Recommended Actions")

        for strategy in strategies:
            st.write("• " + strategy)

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Edit Assessment",
                use_container_width=True
            ):
                st.session_state.page = "Awareness Assessment"
                st.rerun()

        with col2:

            if st.button(
                "New Assessment",
                use_container_width=True
            ):

                for key in list(st.session_state.keys()):

                    if key.startswith("answer_"):
                        del st.session_state[key]

                st.session_state.answers = {}
                st.session_state.prediction = None
                st.session_state.probabilities = None
                st.session_state.page = "Awareness Assessment"

                st.rerun()

        report_text = f"""
CYBERSECURITY AWARENESS ASSESSMENT REPORT

Predicted Awareness Level: {level}
Prediction Confidence: {confidence:.2f}%

Probability Distribution:

{probability_df.to_string(index=False)}

Recommended Actions:

{chr(10).join("- " + x for x in strategies)}
"""

        st.download_button(
            "Download Assessment Report",
            data=report_text,
            file_name="cybersecurity_awareness_report.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================================
# AI INSIGHTS
# ============================================================

elif st.session_state.page == "AI Insights":

    st.markdown(
        '<div class="main-title">AI Insights</div>',
        unsafe_allow_html=True
    )

    prediction = st.session_state.prediction

    if prediction is None:

        st.info(
            "Complete an assessment to view personalised insights."
        )

    else:

        level = get_prediction_category(prediction)

        st.markdown(
            f"""
            <div class="info-card">
            <h3>Assessment Profile</h3>
            <p>
            Your assessment currently indicates a
            <strong>{level}</strong> cybersecurity awareness level.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Security Areas")

        if level == "Low":

            st.write(
                "The assessment suggests that additional "
                "attention should be given to fundamental "
                "cybersecurity practices."
            )

        elif level == "Medium":

            st.write(
                "The assessment suggests a reasonable level "
                "of cybersecurity awareness with opportunities "
                "for improvement."
            )

        else:

            st.write(
                "The assessment suggests strong awareness "
                "of common cybersecurity practices."
            )

        st.subheader("Priority Improvements")

        for item in get_strategy(level):
            st.write("• " + item)


# ============================================================
# ASSESSMENT RECORDS
# ============================================================

elif st.session_state.page == "Assessment Records":

    st.markdown(
        '<div class="main-title">Assessment Records</div>',
        unsafe_allow_html=True
    )

    records = st.session_state.records

    if not records:

        st.info(
            "No assessment records are available in this session."
        )

    else:

        records_df = pd.DataFrame(records)

        search = st.text_input(
            "Search assessment records"
        )

        if search:

            records_df = records_df[
                records_df.astype(str)
                .apply(
                    lambda row: row.str.contains(
                        search,
                        case=False,
                        na=False
                    ).any(),
                    axis=1
                )
            ]

        filter_level = st.selectbox(
            "Filter by awareness level",
            [
                "All",
                "Low",
                "Medium",
                "High"
            ]
        )

        if filter_level != "All":

            records_df = records_df[
                records_df["Prediction"]
                .apply(get_prediction_category)
                == filter_level
            ]

        st.dataframe(
            records_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = records_df.to_csv(index=False)

        st.download_button(
            "Download Records",
            data=csv_data,
            file_name="assessment_records.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "Clear Assessment Records",
            use_container_width=True
        ):

            st.session_state.records = []

            st.rerun()


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

elif st.session_state.page == "Analytics Dashboard":

    st.markdown(
        '<div class="main-title">Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    records = st.session_state.records

    if not records:

        st.info(
            "Complete assessments to populate the analytics dashboard."
        )

    else:

        records_df = pd.DataFrame(records)

        records_df["Level"] = records_df[
            "Prediction"
        ].apply(get_prediction_category)

        total = len(records_df)

        low_count = int(
            (records_df["Level"] == "Low").sum()
        )

        medium_count = int(
            (records_df["Level"] == "Medium").sum()
        )

        high_count = int(
            (records_df["Level"] == "High").sum()
        )

        average_confidence = records_df[
            "Confidence"
        ].mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Assessments",
                total
            )

        with col2:
            st.metric(
                "Low",
                low_count
            )

        with col3:
            st.metric(
                "Medium",
                medium_count
            )

        with col4:
            st.metric(
                "High",
                high_count
            )

        st.metric(
            "Average Confidence",
            f"{average_confidence:.2f}%"
        )

        distribution = pd.DataFrame({
            "Awareness Level": [
                "Low",
                "Medium",
                "High"
            ],
            "Count": [
                low_count,
                medium_count,
                high_count
            ]
        })

        fig = px.bar(
            distribution,
            x="Awareness Level",
            y="Count",
            text="Count"
        )

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )


# ============================================================
# MODEL DASHBOARD
# ============================================================

elif st.session_state.page == "Model Dashboard":

    st.markdown(
        '<div class="main-title">Model Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Overview of the machine-learning model used by the assessment platform.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">Random Forest</div>
                <div class="metric-label">Algorithm</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">68.03%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">69.14%</div>
                <div class="metric-label">5-Fold CV Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">11</div>
                <div class="metric-label">Predictor Variables</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    # --------------------------------------------------------
    # ALGORITHM
    # --------------------------------------------------------

    st.subheader("Algorithm")

    st.markdown(
        """
        <div class="info-card">
        <h3>Random Forest Classifier</h3>
        <p>
        The final cybersecurity awareness prediction model uses a
        Random Forest classification algorithm to classify respondents
        into three awareness levels:
        <strong>Low</strong>, <strong>Medium</strong> and
        <strong>High</strong>.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Model Performance")

    performance_col1, performance_col2 = st.columns(2)

    with performance_col1:

        st.markdown(
            """
            <div class="info-card">
            <h3>Test Set</h3>
            <p>
            <strong>Accuracy:</strong> 68.03%
            </p>
            <p>
            The model correctly classified approximately 68%
            of observations in the held-out test set.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with performance_col2:

        st.markdown(
            """
            <div class="info-card">
            <h3>Cross-Validation</h3>
            <p>
            <strong>5-Fold CV:</strong> 69.14% ± 2.70%
            </p>
            <p>
            The cross-validation results indicate relatively
            consistent performance across the five folds.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Five-Fold Cross-Validation Results")

    cv_df = pd.DataFrame({
        "Fold": [
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5"
        ],
        "Accuracy (%)": [
            69.67,
            67.21,
            69.67,
            65.57,
            73.55
        ]
    })

    cv_fig = px.bar(
        cv_df,
        x="Fold",
        y="Accuracy (%)",
        text="Accuracy (%)"
    )

    cv_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    cv_fig.update_layout(
        template="plotly_white",
        yaxis_title="Accuracy (%)",
        xaxis_title="Cross-Validation Fold",
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(
        cv_fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.subheader("Model Configuration")

    configuration_df = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Number of Trees",
            "Minimum Samples per Leaf",
            "Minimum Samples per Split",
            "Random State",
            "Number of Classes"
        ],
        "Value": [
            "Random Forest Classifier",
            "500",
            "2",
            "4",
            "42",
            "3"
        ]
    })

    st.dataframe(
        configuration_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader("Dataset and Evaluation Split")

    dataset_col1, dataset_col2, dataset_col3, dataset_col4 = st.columns(4)

    with dataset_col1:
        st.metric(
            "Original Records",
            "616"
        )

    with dataset_col2:
        st.metric(
            "Final Modelling Records",
            "609"
        )

    with dataset_col3:
        st.metric(
            "Training Records",
            "487"
        )

    with dataset_col4:
        st.metric(
            "Test Records",
            "122"
        )

    # --------------------------------------------------------
    # PREDICTORS
    # --------------------------------------------------------

    st.subheader("Predictor Variables")

    predictor_df = pd.DataFrame({
        "No.": range(1, len(EXPECTED_FEATURES) + 1),
        "Predictor": [
            DISPLAY_LABELS.get(
                feature,
                feature
            )
            for feature in EXPECTED_FEATURES
        ]
    })

    st.dataframe(
        predictor_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MODEL INTERPRETATION
    # --------------------------------------------------------

    st.subheader("Performance Interpretation")

    st.write(
        "The final Random Forest model achieved 68.03% accuracy "
        "on the held-out test set and a 5-fold cross-validation "
        "accuracy of 69.14% ± 2.70%."
    )

    st.write(
        "The model performs most strongly for the Medium awareness "
        "class, while performance for the Low and High classes is "
        "more limited. These results should therefore be interpreted "
        "as model-based estimates rather than absolute measures of "
        "an individual's cybersecurity awareness."
    )

    # --------------------------------------------------------
    # DATA LEAKAGE PREVENTION
    # --------------------------------------------------------

    st.subheader("Data Leakage Prevention")

    st.write(
        "The cybersecurity-practice questions used to construct "
        "the Awareness_Level target were excluded from the model "
        "predictors. The final model therefore uses 11 independent "
        "predictor variables rather than directly using the target "
        "questions as input."
    )


# ============================================================
# CYBERSECURITY GUIDANCE
# ============================================================

elif st.session_state.page == "Cybersecurity Guidance":

    st.markdown(
        '<div class="main-title">Cybersecurity Guidance</div>',
        unsafe_allow_html=True
    )

    guidance = {
        "Strong Passwords":
            "Use long, unique passwords and avoid reusing passwords across accounts.",

        "Phishing":
            "Check links, attachments, senders and requests for sensitive information before responding.",

        "Device Security":
            "Use security software and enable appropriate device security controls.",

        "Software Updates":
            "Install operating system, application and security updates regularly.",

        "Public Wi-Fi":
            "Avoid sensitive activities on unsecured public networks and use trusted connections when possible.",

        "Privacy":
            "Review privacy settings and limit the amount of personal information shared online.",

        "Safe Browsing":
            "Use trusted websites, check website addresses and avoid suspicious downloads."
    }

    for topic, description in guidance.items():

        st.markdown(
            f"""
            <div class="info-card">
            <h3>{topic}</h3>
            <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FEEDBACK
# ============================================================

elif st.session_state.page == "Feedback":

    st.markdown(
        '<div class="main-title">Feedback</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Share your experience with the cybersecurity awareness assessment."
    )

    rating = st.slider(
        "How would you rate the assessment?",
        min_value=1,
        max_value=5,
        value=5
    )

    review_text = st.text_area(
        "Your feedback",
        placeholder="Write your feedback here..."
    )

    if st.button(
        "Submit Feedback",
        use_container_width=True
    ):

        if not review_text.strip():

            st.warning(
                "Please enter feedback before submitting."
            )

        else:

            new_review = {
                "rating": rating,
                "review": review_text.strip(),
                "date": datetime.now().strftime(
                    "%d %B %Y • %H:%M"
                )
            }

            st.session_state.reviews.insert(
                0,
                new_review
            )

            st.success(
                "Thank you for your feedback!"
            )

    st.subheader("Recent Feedback")

    reviews = st.session_state.reviews

    if not reviews:

        st.info(
            "No feedback has been submitted yet."
        )

    else:

        for item in reviews:

            stars = "⭐" * int(
                item.get("rating", 0)
            )

            st.subheader(stars)

            st.write(
                item.get("review", "")
            )

            st.caption(
                "Anonymous reviewer • "
                + item.get("date", "")
            )

            st.divider()


# ============================================================
# ABOUT & METHODOLOGY
# ============================================================

elif st.session_state.page == "About & Methodology":

    st.markdown(
        '<div class="main-title">About & Methodology</div>',
        unsafe_allow_html=True
    )

    st.subheader("Research Title")

    st.write(
        "A Machine Learning-Based Approach to Predict "
        "Cybersecurity Awareness for University Students."
    )

    st.subheader("Objective")

    st.write(
        "The objective is to develop a machine-learning-based "
        "approach for predicting cybersecurity awareness levels "
        "using selected characteristics, digital behaviours, "
        "security practices and cybersecurity perceptions."
    )

    st.subheader("Dataset")

    st.write(
        "The research dataset originally contained 616 records "
        "and 63 variables. After duplicate removal and preparation "
        "of valid target records, 609 records were used for modelling."
    )

    st.subheader("Target Variable")

    st.write(
        "The target variable is Awareness_Level, classified into "
        "Low, Medium and High awareness categories."
    )

    st.subheader("Predictor Variables")

    for feature in EXPECTED_FEATURES:

        st.write(
            "• " + DISPLAY_LABELS.get(
                feature,
                feature
            )
        )

    st.subheader("Machine Learning Algorithm")

    st.write(
        "The final prediction algorithm is a Random Forest Classifier "
        "with 500 trees, minimum samples per leaf of 2, minimum samples "
        "per split of 4, and random state 42."
    )

    st.subheader("Evaluation")

    st.write(
        "The dataset was divided into 80% training data and 20% "
        "test data using stratified sampling. The model achieved "
        "68.03% test accuracy and 69.14% ± 2.70% five-fold "
        "cross-validation accuracy."
    )

    st.subheader("Data Leakage Prevention")

    st.write(
        "The questions used to construct the Awareness_Level target "
        "were excluded from the predictor variables to reduce the "
        "risk of target leakage."
    )

    st.subheader("Limitations")

    st.write(
        "The model does not perform equally across all "
        "awareness categories. Individual predictions should "
        "therefore be interpreted as model-based estimates."
    )

    st.subheader("Privacy")

    st.write(
        "The assessment does not require a person's name, "
        "student number or email address to generate a prediction."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cybersecurity Awareness Assessment • "
    "Machine-learning based assessment platform"
)
