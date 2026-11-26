import os
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from supabase import create_client, Client

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ID = "RioCH/stratified_sample"
REPO_FILE = "stratified_evaluation_250.parquet"

CRITERIA = {
    "Relevance": "Is the generated question relevant to the provided context and target answer?",
    "Answerability": "Can the target answer be correctly inferred from the generated question and the provided context?",
    "Fluency": "Is the generated question grammatically correct, clear, and natural?",
    "Factual_Consistency": "Is the generated question factually consistent with the context, without introducing unsupported information?",
}

st.set_page_config(page_title="AQG Human Annotation", page_icon="📝", layout="wide")



st.markdown("""
<style>
/* Increase font size of all four annotation content boxes */
div[data-testid="stTextArea"] textarea {
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    opacity: 1 !important;
}

/* Preserve white text in disabled/read-only text areas */
div[data-testid="stTextArea"] textarea:disabled {
    color: white !important;
    -webkit-text-fill-color: white !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)




# ============================================================
# CONFIG / DATABASE
# ============================================================
def get_secret(section, key, default=None):
    try:
        return st.secrets[section][key]
    except Exception:
        return default


def get_annotators():
    try:
        names = list(st.secrets["annotators"])
    except Exception:
        names = ["Rio", "Ryan", "Jericho"]
    names = [str(x).strip() for x in names if str(x).strip()]
    if not names:
        raise ValueError("Configure at least one name in .streamlit/secrets.toml.")
    return names


@st.cache_resource
def get_supabase() -> Client:
    url = get_secret("supabase", "url")
    key = get_secret("supabase", "service_role_key")
    if not url or not key:
        raise RuntimeError(
            "Supabase credentials are missing. Configure supabase.url and "
            "supabase.service_role_key in Streamlit secrets."
        )
    return create_client(url, key)


def fetch_annotator_annotations(annotator_id):
    response = (
        get_supabase()
        .table("annotations")
        .select("*")
        .eq("annotator_id", annotator_id)
        .order("Sample_ID")
        .execute()
    )
    return pd.DataFrame(response.data or [])


def fetch_all_annotations():
    response = (
        get_supabase()
        .table("annotations")
        .select("*")
        .order("annotator_id")
        .order("Sample_ID")
        .execute()
    )
    return pd.DataFrame(response.data or [])


def save_annotation(annotator_id, Sample_ID, values):
    row = {
        "annotator_id": annotator_id,
        "Sample_ID": int(Sample_ID),
        "relevance": values["Relevance"],
        "answerability": values["Answerability"],
        "fluency": values["Fluency"],
        "factual_consistency": values["Factual_Consistency"],
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (
        get_supabase()
        .table("annotations")
        .upsert(row, on_conflict="annotator_id,Sample_ID")
        .execute()
    )


def row_to_annotation(row):
    if row is None:
        return None
    return {
        "Relevance": row.get("relevance", ""),
        "Answerability": row.get("answerability", ""),
        "Fluency": row.get("fluency", ""),
        "Factual_Consistency": row.get("factual_consistency", ""),
        "Updated_At": row.get("updated_at", ""),
    }


def is_complete_row(row):
    if row is None:
        return False
    return all(
        str(row.get(col, "")) in ("True", "False")
        for col in ("relevance", "answerability", "fluency", "factual_consistency")
    )


# ============================================================
# DATA LOADING (same dataset and answer-matching logic)
# ============================================================
@st.cache_data(show_spinner="Loading stratified samples and retrieving NLQuAD answers...")
def load_samples():
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HF_TOKEN"))

    stratified_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=REPO_FILE,
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(stratified_path)

    required_sample_cols = ["Sample_ID", "Stratum", "Context", "Reference", "Prediction"]
    missing = [col for col in required_sample_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in stratified sample: {missing}")

    df["Sample_ID"] = df["Sample_ID"].astype(int)
    df = df.sort_values("Sample_ID").reset_index(drop=True)
    if df["Sample_ID"].duplicated().any():
        raise ValueError("Duplicate Sample_ID detected.")

    source_path = hf_hub_download(
        repo_id="Yoga26/new_nlquad_filtered",
        filename="eval.parquet",
        repo_type="dataset",
        token=token,
    )
    source_df = pd.read_parquet(source_path)

    required_source_cols = ["context", "question", "answer_text"]
    missing = [col for col in required_source_cols if col not in source_df.columns]
    if missing:
        raise ValueError(f"Missing columns in NLQuAD dataset: {missing}")

    def normalize_text(value):
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    source_df = source_df.copy()
    source_df["context_key"] = source_df["context"].map(normalize_text)
    source_df["question_key"] = source_df["question"].map(normalize_text)
    df["context_key"] = df["Context"].map(normalize_text)
    df["question_key"] = df["Reference"].map(normalize_text)

    answer_counts = source_df.groupby(
        ["context_key", "question_key"]
    )["answer_text"].nunique(dropna=False)
    ambiguous_pairs = answer_counts[answer_counts > 1]
    if not ambiguous_pairs.empty:
        raise ValueError(
            f"Found {len(ambiguous_pairs)} conflicting Context + Reference answer pairs."
        )

    answer_lookup = (
        source_df[["context_key", "question_key", "answer_text"]]
        .drop_duplicates(subset=["context_key", "question_key"])
        .rename(columns={"answer_text": "Answer"})
    )

    df["_original_order"] = range(len(df))
    df = df.merge(
        answer_lookup,
        on=["context_key", "question_key"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    df = (
        df.sort_values("_original_order")
        .drop(columns=["context_key", "question_key", "_original_order"])
        .reset_index(drop=True)
    )

    missing_answers = df["Answer"].isna()
    if missing_answers.any():
        missing_ids = df.loc[missing_answers, "Sample_ID"].tolist()
        raise ValueError(f"Could not retrieve answers for Sample_ID: {missing_ids}")

    if len(df) != 250 or not df["Sample_ID"].is_unique:
        raise ValueError(
            f"Expected 250 unique samples, found {len(df)} rows."
        )
    return df


# ============================================================
# APP
# ============================================================
st.title("📝 AQG Human Annotation Viewer")
st.caption(
    "Human evaluation interface for the 250 stratified T5 question-generation samples."
)

try:
    annotators = get_annotators()
    supabase = get_supabase()
except Exception as exc:
    st.error(f"App configuration error: {exc}")
    st.info(
        "Set up Supabase and configure the Streamlit secrets as described in "
        "the deployment README."
    )
    st.stop()

# Dropdown identity. This records the selected label; it is not authentication.
if "annotator_id" not in st.session_state:
    st.session_state.annotator_id = annotators[0]

with st.sidebar:
    st.header("Annotator")
    selected_annotator = st.selectbox(
        "Select your annotator ID",
        options=annotators,
        index=annotators.index(st.session_state.annotator_id)
        if st.session_state.annotator_id in annotators else 0,
        key="annotator_selector",
    )
    if selected_annotator != st.session_state.annotator_id:
        st.session_state.annotator_id = selected_annotator
        st.session_state.pop("sample_index", None)
        st.rerun()

annotator_id = st.session_state.annotator_id

try:
    samples = load_samples()
    annotation_df = fetch_annotator_annotations(annotator_id)
except Exception as exc:
    st.error(f"Could not load the samples or annotations: {exc}")
    st.stop()

total = len(samples)
Sample_IDs = samples["Sample_ID"].astype(int).tolist()

# Map this annotator's own saved annotations by sample ID.
saved_by_id = {}
if not annotation_df.empty:
    for row in annotation_df.to_dict("records"):
        saved_by_id[int(row["Sample_ID"])] = row

completed_ids = {
    sid for sid, row in saved_by_id.items() if is_complete_row(row)
}
completed = len(completed_ids)

if "sample_index" not in st.session_state:
    first_incomplete = next(
        (i for i, sid in enumerate(Sample_IDs) if sid not in completed_ids),
        max(total - 1, 0),
    )
    st.session_state.sample_index = first_incomplete
    st.session_state.sample_selector = Sample_IDs[first_incomplete]


def select_sample_changed():
    selected = int(st.session_state.sample_selector)
    st.session_state.sample_index = Sample_IDs.index(selected)


def go_previous():
    new_index = max(0, int(st.session_state.sample_index) - 1)
    st.session_state.sample_index = new_index
    st.session_state.sample_selector = Sample_IDs[new_index]


def go_next():
    new_index = min(total - 1, int(st.session_state.sample_index) + 1)
    st.session_state.sample_index = new_index
    st.session_state.sample_selector = Sample_IDs[new_index]


current_index = min(max(int(st.session_state.sample_index), 0), total - 1)
current = samples.iloc[current_index]
Sample_ID = int(current["Sample_ID"])
saved = row_to_annotation(saved_by_id.get(Sample_ID))

# Sidebar progress and navigation
with st.sidebar:
    st.divider()
    st.header("Annotation progress")
    st.metric("Completed samples", f"{completed} / {total}")
    st.progress(completed / total if total else 0)
    st.caption(f"Remaining: {total - completed}")

    st.divider()
    st.subheader("Navigate to sample")
    st.selectbox(
        "Sample ID",
        options=Sample_IDs,
        index=current_index,
        key="sample_selector",
        on_change=select_sample_changed,
    )

    st.caption("Annotation status")
    st.write("✅ Annotated" if saved else "⬜ Not yet annotated")

    st.divider()
    st.subheader("Filter by stratum")
    strata = ["All"] + sorted(samples["Stratum"].dropna().astype(str).unique().tolist())
    selected_stratum = st.selectbox("Stratum", strata)
    if selected_stratum != "All":
        stratum_ids = samples.loc[
            samples["Stratum"].astype(str) == selected_stratum, "Sample_ID"
        ].tolist()
        st.caption(f"{len(stratum_ids)} samples in {selected_stratum}")
        st.caption("Use the Sample ID selector above to jump to a sample in this stratum.")

# Sample header
col_a, col_b, col_c = st.columns(3)
col_a.metric("Current sample", f"{current_index + 1} / {total}")
col_b.metric("Sample ID", Sample_ID)
col_c.metric("Stratum", str(current["Stratum"]))
st.progress((current_index + 1) / total)

# Main sample content — single-column layout

st.subheader("Target Answer")
st.text_area(
    "Answer from NLQuAD",
    value=str(current["Answer"]),
    height=180,
    disabled=True,
    key=f"answer_{Sample_ID}"
)

st.subheader("Reference Question")
st.text_area(
    "Original question",
    value=str(current["Reference"]),
    height=150,
    disabled=True,
    key=f"reference_{Sample_ID}"
)

st.subheader("Generated Question")
st.text_area(
    "T5 prediction",
    value=str(current["Prediction"]),
    height=150,
    disabled=True,
    key=f"prediction_{Sample_ID}"
)

st.subheader("Context")
st.text_area(
    "Original passage",
    value=str(current["Context"]),
    height=500,
    disabled=True,
    key=f"context_{Sample_ID}"
)

st.divider()
st.header("Human Annotation")
st.caption(
    "Evaluate the generated question using the four criteria below. "
    "Select True or False for every criterion, then save."
)

saved_values = saved or {}
options = ["Not annotated", "True", "False"]
if saved:
    st.info(
        "This sample has a saved annotation for your selected annotator. "
        "Its previous judgments are pre-filled; save again to update them."
    )

with st.form(key=f"annotation_form_{annotator_id}_{Sample_ID}"):
    selections = {}
    for criterion, description in CRITERIA.items():
        st.markdown(f"**{criterion.replace('_', ' ')}**")
        st.caption(description)
        previous_value = str(saved_values.get(criterion, ""))
        default_value = (
            previous_value if previous_value in ("True", "False") else "Not annotated"
        )
        selections[criterion] = st.radio(
            label=f"{criterion} judgment",
            options=options,
            index=options.index(default_value),
            horizontal=True,
            key=f"{annotator_id}_{criterion}_{Sample_ID}",
            label_visibility="collapsed",
        )
        st.write("")

    submitted = st.form_submit_button(
        "💾 Save Annotation", type="primary", use_container_width=True
    )

if submitted:
    if any(value == "Not annotated" for value in selections.values()):
        st.warning("Please select True or False for all four criteria before saving.")
    else:
        try:
            save_annotation(annotator_id, Sample_ID, selections)
            st.success(
                f"Annotation for Sample_ID {Sample_ID} saved for {annotator_id}."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save annotation to Supabase: {exc}")

# Navigation controls
st.divider()
nav_prev, nav_status, nav_next = st.columns([1, 2, 1])
with nav_prev:
    st.button("← Previous", disabled=(current_index == 0),
              use_container_width=True, on_click=go_previous)
with nav_status:
    st.markdown(
        f"<div style='text-align:center;padding-top:8px;'>"
        f"Sample {current_index + 1} of {total}</div>",
        unsafe_allow_html=True,
    )
with nav_next:
    st.button("Next →", disabled=(current_index >= total - 1),
              use_container_width=True, on_click=go_next)

# Export current annotator's CSV
st.sidebar.divider()
st.sidebar.subheader("Export")
if not annotation_df.empty:
    own_export = annotation_df.copy()
    own_export = own_export.rename(columns={
        "annotator_id": "Annotator_ID",
        "Sample_ID": "Sample_ID",
        "relevance": "Relevance",
        "answerability": "Answerability",
        "fluency": "Fluency",
        "factual_consistency": "Factual_Consistency",
        "updated_at": "Updated_At",
    })
    own_export = own_export[
        ["Annotator_ID", "Sample_ID", "Relevance", "Answerability",
         "Fluency", "Factual_Consistency", "Updated_At"]
    ]
    st.sidebar.download_button(
        "Download My Annotations",
        data=own_export.to_csv(index=False).encode("utf-8"),
        file_name=f"annotations_{annotator_id.replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.sidebar.caption("Your CSV will be available after your first saved annotation.")

# Optional administrator export: enabled only when admin_password is configured.
admin_password = get_secret("admin_password", "value", "")
if admin_password:
    with st.sidebar.expander("Administrator export"):
        entered_password = st.text_input("Admin password", type="password")
        if st.button("Load all annotators' data", use_container_width=True):
            if entered_password == admin_password:
                try:
                    all_df = fetch_all_annotations()
                    st.session_state["admin_export_ready"] = True
                    st.session_state["admin_export_df"] = all_df
                except Exception as exc:
                    st.error(f"Could not retrieve all annotations: {exc}")
            else:
                st.error("Incorrect admin password.")

        if st.session_state.get("admin_export_ready", False):
            all_df = st.session_state.get("admin_export_df", pd.DataFrame())
            if not all_df.empty:
                all_export = all_df.rename(columns={
                    "annotator_id": "Annotator_ID",
                    "Sample_ID": "Sample_ID",
                    "relevance": "Relevance",
                    "answerability": "Answerability",
                    "fluency": "Fluency",
                    "factual_consistency": "Factual_Consistency",
                    "updated_at": "Updated_At",
                })
                all_export = all_export[
                    ["Annotator_ID", "Sample_ID", "Relevance", "Answerability",
                     "Fluency", "Factual_Consistency", "Updated_At"]
                ]
                st.download_button(
                    "Download All Annotations CSV",
                    data=all_export.to_csv(index=False).encode("utf-8"),
                    file_name="all_human_annotations.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No annotation records have been saved yet.")

st.sidebar.caption(
    "Annotations are stored centrally in Supabase. Each annotator's records "
    "are kept separate by the (Annotator_ID, Sample_ID) pair."
)
