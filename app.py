import streamlit as st
import pandas as pd
import numpy as np
import hnswlib
import os
import pickle
import ast

processed_dir = "processed"


@st.cache_resource
def load_all_data():
    with open(os.path.join(processed_dir, "lightfm_model.pkl"), "rb") as f:
        model = pickle.load(f)

    reviews = pd.read_csv(
        os.path.join(processed_dir, "processed_reviews.csv"), index_col=0
    )
    metadata = pd.read_csv(
        os.path.join(processed_dir, "processed_metadata.csv"), index_col=0
    )

    with open(os.path.join(processed_dir, "user_to_id_map.pkl"), "rb") as f:
        user_to_id = pickle.load(f)

    with open(os.path.join(processed_dir, "item_to_id_map.pkl"), "rb") as f:
        item_to_id = pickle.load(f)

    with open(os.path.join(processed_dir, "lightfm_idx_to_item_id.pkl"), "rb") as f:
        idx_to_item_id = pickle.load(f)

    item_embeddings = np.load(os.path.join(processed_dir, "item_embeddings.npy"))
    item_biases = np.load(os.path.join(processed_dir, "item_biases.npy"))

    dim = item_embeddings.shape[1]
    num_elements = item_embeddings.shape[0]
    hnsw_index = hnswlib.Index(space="ip", dim=dim)
    hnsw_index.load_index(
        os.path.join(processed_dir, "hnsw_index.bin"), max_elements=num_elements
    )
    hnsw_index.set_ef(100)

    return (
        model,
        reviews,
        metadata,
        user_to_id,
        item_to_id,
        idx_to_item_id,
        item_embeddings,
        item_biases,
        hnsw_index,
    )


def get_recommendations(
    user_id,
    model,
    reviews_df,
    metadata_df,
    user_to_id_map,
    item_to_id_map,
    idx_to_item_id,
    hnsw_index,
    item_embeddings,
    item_biases,
    num_recommendations=10,
    k_candidates_multiplier=5,
):

    user_idx = user_to_id_map.get(user_id)
    if user_idx is None:
        return []

    user_emb = model.user_embeddings[user_idx]
    user_bias = model.user_biases[user_idx]

    k = num_recommendations * k_candidates_multiplier
    if k == 0:
        k = num_recommendations

    candidate_idxs, _ = hnsw_index.knn_query(user_emb, k=k)
    candidate_idxs = candidate_idxs[0]

    seen_asins = reviews_df[reviews_df["user_id"] == user_id]["asin"].unique()
    seen_idxs = {item_to_id_map[a] for a in seen_asins if a in item_to_id_map}

    results = []
    for idx in candidate_idxs:
        if idx in seen_idxs:
            continue
        emb = item_embeddings[idx]
        bias = item_biases[idx]
        score = np.dot(user_emb, emb) + user_bias + bias

        asin = idx_to_item_id.get(idx)
        item_row = metadata_df[metadata_df["asin"] == asin]
        if not item_row.empty:
            title = item_row["title"].iloc[0] if "title" in item_row.columns else "N/A"
            categories = (
                item_row["categories"].iloc[0]
                if "categories" in item_row.columns
                else "N/A"
            )
            categories = ", ".join(ast.literal_eval(categories))
            average_rating = item_row["average_rating"].iloc[0]
            rating_number = item_row["rating_number"].iloc[0]
            results.append(
                {
                    "item_id": asin,
                    "title": title,
                    "categories": categories,
                    "average_rating": average_rating,
                    "rating_number": rating_number,
                    "predicted_score": score,
                }
            )
            if len(results) >= num_recommendations:
                break

    return sorted(results, key=lambda x: x["predicted_score"], reverse=True)


st.set_page_config(page_title="Amazon Recommender", layout="wide")
st.title("Amazon Video Game Recommendation System")

st.markdown(
    """
This app shows past user reviews and generates recommendations using a hybrid LightFM + HNSW system.
"""
)

(
    model,
    reviews_df,
    metadata_df,
    user_to_id,
    item_to_id,
    idx_to_item_id,
    item_embeddings,
    item_biases,
    hnsw_index,
) = load_all_data()

unique_users = reviews_df["user_id"].unique()

if "selected_user" not in st.session_state:
    st.session_state.selected_user = np.random.choice(unique_users)

if st.button("Refresh with Random User"):
    st.session_state.selected_user = np.random.choice(unique_users)

user_id = st.selectbox(
    "Select a User ID",
    unique_users,
    index=int(np.where(unique_users == st.session_state.selected_user)[0][0]),
)

st.subheader("User Review History")
user_reviews = reviews_df[reviews_df["user_id"] == user_id]
user_items = metadata_df[metadata_df["asin"].isin(user_reviews["asin"].unique())]
history = user_reviews.merge(user_items, on="asin", how="left")

if not history.empty:
    st.dataframe(
        history[
            [
                "title_y",
                "main_category_y",
                "title_x",
                "rating",
                "verified_purchase",
            ]
        ].rename(
            columns={
                "title_y": "Item Title",
                "main_category_y": "Main Category",
                "title_x": "Review Title",
                "rating": "Rating",
                "verified_purchase": "Verified Purchase",
            }
        )
    )
else:
    st.warning("This user has no matching item metadata.")

st.subheader("Recommended Items")
recs = get_recommendations(
    user_id,
    model,
    reviews_df,
    metadata_df,
    user_to_id,
    item_to_id,
    idx_to_item_id,
    hnsw_index,
    item_embeddings,
    item_biases,
)

if recs:
    rec_df = pd.DataFrame(recs)[
        ["title", "categories", "average_rating", "rating_number", "predicted_score"]
    ].rename(
        columns={
            "title": "Title",
            "categories": "Categories",
            "average_rating": "Average Rating",
            "rating_number": "Rating Number",
            "predicted_score": "Score",
        }
    )
    st.dataframe(rec_df)
else:
    st.info("No recommendations available for this user.")
