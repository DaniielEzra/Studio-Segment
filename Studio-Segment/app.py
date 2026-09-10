import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


load_dotenv()


def detect_id_columns(df: pd.DataFrame):
    """Identify only clearly identifying columns that should not be used for clustering."""
    ignored = set()

    for column in df.columns:
        column_name = str(column).strip()
        normalized_name = (
            column_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        )

        exact_identifier_names = {
            "id",
            "uuid",
            "identifier",
            "index",
            "record_id",
            "recordid",
            "row_id",
            "rowid",
            "unique_id",
            "uniqueid",
        }

        clear_identifier_patterns = (
            normalized_name in exact_identifier_names
            or normalized_name.endswith("_id")
            or normalized_name.endswith("_uuid")
            or normalized_name.endswith("_identifier")
            or normalized_name.endswith("_record_id")
            or normalized_name.endswith("_row_id")
        )

        if clear_identifier_patterns:
            ignored.add(column)
            continue

        values = df[column]
        valid_values = values.dropna()
        if len(valid_values) == 0:
            continue

        unique_ratio = valid_values.nunique() / len(valid_values)
        if unique_ratio > 0.95 and normalized_name not in {"customer", "user", "employee"}:
            ignored.add(column)

    return sorted(ignored)


def get_feature_columns(df: pd.DataFrame):
    """Return usable columns for clustering after removing obvious ID columns."""
    ignored_columns = detect_id_columns(df)
    usable_columns = [col for col in df.columns if col not in ignored_columns]

    if not usable_columns:
        raise ValueError("No usable columns remain for clustering after removing ID-like fields.")

    numeric_columns = [
        col for col in usable_columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_columns = [
        col for col in usable_columns if col not in numeric_columns
    ]

    return {
        "all": usable_columns,
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "ignored": ignored_columns,
    }


def prepare_data_for_clustering(df: pd.DataFrame):
    """Clean data, encode categories, and scale the values for K-Means."""
    feature_info = get_feature_columns(df)
    numerical_cols = feature_info["numeric"]
    categorical_cols = feature_info["categorical"]

    if not numerical_cols and not categorical_cols:
        raise ValueError(
            "The dataset contains no usable numerical or categorical features for clustering."
        )

    selected_df = df[feature_info["all"]].copy()

    # Fill missing values before modeling.
    for col in numerical_cols:
        selected_df[col] = selected_df[col].fillna(selected_df[col].median())

    for col in categorical_cols:
        selected_df[col] = selected_df[col].fillna(selected_df[col].mode(dropna=True).iloc[0])

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numerical_cols:
        transformers.append(("num", numerical_transformer, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    X = preprocessor.fit_transform(selected_df)

    return {
        "df_clean": selected_df,
        "X": X,
        "preprocessor": preprocessor,
        "feature_info": feature_info,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }


def run_wcss_analysis(X: np.ndarray, min_k: int, max_k: int):
    """Run K-Means for each K in the selected range and calculate WCSS."""
    if min_k < 1 or max_k < 1:
        raise ValueError("K values must be at least 1.")

    if min_k > max_k:
        raise ValueError("The minimum K must be smaller than or equal to the maximum K.")

    wcss_results = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        wcss_results.append({"k": k, "wcss": float(model.inertia_)})

    return pd.DataFrame(wcss_results)


def get_openai_api_key():
    """Read the OpenAI API key from environment variables or Streamlit secrets."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key. Add OPENAI_API_KEY to your environment or Streamlit secrets.")
    return api_key


def summarize_cluster(cluster_df: pd.DataFrame, numeric_cols, categorical_cols):
    """Create a compact textual summary for one cluster from the raw data."""
    summary = {
        "count": int(len(cluster_df)),
        "numeric_averages": {},
        "categorical_modes": {},
    }

    for col in numeric_cols:
        if col in cluster_df.columns and not cluster_df[col].dropna().empty:
            summary["numeric_averages"][col] = round(float(cluster_df[col].mean()), 3)

    for col in categorical_cols:
        if col in cluster_df.columns:
            non_null = cluster_df[col].dropna()
            if not non_null.empty:
                summary["categorical_modes"][col] = non_null.mode().iloc[0]
            else:
                summary["categorical_modes"][col] = "missing"

    return summary


def explain_cluster_with_llm(cluster_summary):
    """Ask the LLM for a short, useful cluster name and description."""
    client = OpenAI(api_key=get_openai_api_key())

    prompt = f"""
    You are a data analyst. Based only on the summary below, suggest a short meaningful name and a one-line description for this cluster.

    Cluster summary:
    {json.dumps(cluster_summary, ensure_ascii=False, indent=2)}

    Return valid JSON with exactly two keys:
    - name: a short name for the cluster
    - description: a one-line description in plain English

    Do not include any extra text outside the JSON object.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful data analyst who names customer or data segments in a concise way.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    name = str(parsed.get("name", "Unnamed segment")).strip()
    description = str(parsed.get("description", "No description available.")).strip()

    return {"name": name, "description": description}


def get_cluster_table(clustered_df: pd.DataFrame):
    """Return a table with cluster count and summary columns."""
    cluster_table = (
        clustered_df.groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    cluster_table["name"] = ""
    cluster_table["description"] = ""
    return cluster_table[["cluster_id", "count", "name", "description"]]


def main():
    """Create the Streamlit application interface and workflow."""
    st.set_page_config(page_title="Studio Segment", page_icon="📊", layout="wide")
    st.title("Studio Segment – Automatic Data Segmentation")
    st.caption("Upload a CSV file, find natural groupings with K-Means, and interpret each segment with an LLM.")

    # Step 1 - Upload CSV
    st.subheader("Step 1 - Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    st.session_state["raw_df"] = raw_df

    col1, col2 = st.columns(2)
    col1.metric("Rows", len(raw_df))
    col2.metric("Columns", len(raw_df.columns))

    st.write("Dataset preview:")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # Step 2 - Preprocessing
    st.subheader("Step 2 - Preprocessing")
    try:
        prep_result = prepare_data_for_clustering(raw_df)
        st.session_state["prep_result"] = prep_result

        feature_info = prep_result["feature_info"]
        st.success("Preprocessing completed successfully.")

        st.write("Ignored columns:")
        if feature_info["ignored"]:
            st.code(", ".join(feature_info["ignored"]))
        else:
            st.write("No obvious ID-like columns were found.")

        st.write("Used for clustering:")
        st.code(", ".join(feature_info["all"]))

        st.write("Numerical columns:")
        st.code(", ".join(feature_info["numeric"]) if feature_info["numeric"] else "None")

        st.write("Categorical columns:")
        st.code(", ".join(feature_info["categorical"]) if feature_info["categorical"] else "None")

        st.markdown(
            """
            - Missing numerical values were filled using the median.
            - Missing categorical values were filled using the most common value.
            - Categorical variables were encoded into numeric values.
            - Numerical features were scaled before K-Means.
            """
        )
    except ValueError as exc:
        st.error(f"Preprocessing failed: {exc}")
        return

    # Step 3 - Elbow Method
    st.subheader("Step 3 - Elbow Method")
    min_k = st.number_input("Minimum K", min_value=1, max_value=max(1, min(20, len(raw_df))), value=2)
    max_k = st.number_input("Maximum K", min_value=1, max_value=max(1, min(20, len(raw_df))), value=min(5, len(raw_df)))

    if min_k > max_k:
        st.error("The minimum K must be less than or equal to the maximum K.")
    else:
        if max_k >= len(raw_df):
            st.warning("The selected K range is large relative to the number of rows. This can produce unstable clustering.")

        if st.button("Calculate WCSS"):
            try:
                wcss_df = run_wcss_analysis(prep_result["X"], min_k, max_k)
                st.session_state["wcss_df"] = wcss_df

                st.write("WCSS results:")
                st.dataframe(wcss_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(wcss_df["k"], wcss_df["wcss"], marker="o")
                ax.set_xlabel("K")
                ax.set_ylabel("WCSS")
                ax.set_title("Elbow Method")
                ax.grid(True)
                st.pyplot(fig)
            except Exception as exc:
                st.error(f"Failed to calculate WCSS: {exc}")

    # Step 4 - Create Clusters
    st.subheader("Step 4 - Create Clusters")
    if "wcss_df" in st.session_state:
        k_options = list(range(int(st.session_state["wcss_df"]["k"].min()), int(st.session_state["wcss_df"]["k"].max()) + 1))
    else:
        k_options = list(range(min_k, max_k + 1))

    final_k = st.selectbox("Choose the final K", options=k_options)

    if final_k >= len(raw_df):
        st.error("The selected K cannot be greater than or equal to the number of rows in the dataset.")
    else:
        if st.button("Create clusters"):
            try:
                model = KMeans(n_clusters=final_k, n_init=10, random_state=42)
                labels = model.fit_predict(prep_result["X"])
                clustered_df = raw_df.copy()
                clustered_df["cluster_id"] = labels

                cluster_counts = (
                    clustered_df.groupby("cluster_id", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                )
                cluster_counts["name"] = ""
                cluster_counts["description"] = ""

                st.session_state["clustered_df"] = clustered_df
                st.session_state["cluster_counts"] = cluster_counts

                st.write("Cluster counts:")
                st.dataframe(cluster_counts[["cluster_id", "count", "name", "description"]], use_container_width=True)
            except Exception as exc:
                st.error(f"Cluster creation failed: {exc}")

    # Step 5 - Explain Clusters with LLM
    st.subheader("Step 5 - Explain Clusters with LLM")
    if "clustered_df" in st.session_state and "cluster_counts" in st.session_state:
        if st.button("Explain clusters with LLM"):
            try:
                cluster_rows = []
                numeric_cols = prep_result["numerical_cols"]
                categorical_cols = prep_result["categorical_cols"]

                for cluster_id, group in st.session_state["clustered_df"].groupby("cluster_id"):
                    summary = summarize_cluster(group, numeric_cols, categorical_cols)
                    llm_result = explain_cluster_with_llm(summary)
                    cluster_rows.append(
                        {
                            "cluster_id": int(cluster_id),
                            "count": int(summary["count"]),
                            "name": llm_result["name"],
                            "description": llm_result["description"],
                        }
                    )

                final_cluster_table = pd.DataFrame(cluster_rows)
                st.session_state["final_cluster_table"] = final_cluster_table

                st.write("Final cluster table:")
                st.dataframe(final_cluster_table[["cluster_id", "count", "name", "description"]], use_container_width=True)
            except ValueError as exc:
                st.error(f"LLM configuration error: {exc}")
            except Exception as exc:
                st.error(f"Cluster explanation failed: {exc}")
    else:
        st.info("Create clusters before explaining them.")

    # Step 6 - Export CSV
    st.subheader("Step 6 - Export CSV")
    if "final_cluster_table" in st.session_state:
        export_df = st.session_state["raw_df"].copy()
        name_map = (
            st.session_state["final_cluster_table"].set_index("cluster_id")["name"].to_dict()
        )

        # Add the final cluster names to the original dataset.
        if "cluster_id" in st.session_state["clustered_df"].columns:
            export_df = st.session_state["clustered_df"].copy()
            export_df["name_cluster"] = export_df["cluster_id"].map(name_map)

            original_name = uploaded_file.name.rsplit(".", 1)[0]
            download_name = f"{original_name}_clustered.csv"

            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Download clustered CSV",
                data=csv_data,
                file_name=download_name,
                mime="text/csv",
            )

            st.write("Export preview:")
            st.dataframe(export_df[["cluster_id", "name_cluster"]].head(20), use_container_width=True)
    else:
        st.info("Create and explain the clusters before exporting a CSV file.")


if __name__ == "__main__":
    main()
