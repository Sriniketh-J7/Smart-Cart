import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#E63946", "#457B9D", "#F4A261", "#2A9D8F"]
CLUSTER_NAMES = {
    0: "Mid-Income Moderate Spenders",
    1: "High-Value Premium Buyers",
    2: "Budget-Conscious Browsers",
    3: "High-Income Campaign Responders",
}
CLUSTER_STRATEGIES = {
    0: {
        "title": "Cluster 0 — Mid-Income Moderate Spenders",
        "insight": "Large middle-income group with more children. High web visits but low conversions — browsers, not buyers.",
        "strategy": [
            "Run discount-led conversion campaigns targeting frequent web visitors.",
            "Promote family product bundles (essentials, sweets, fruits).",
            "Use retargeting ads for customers who visit without purchasing.",
            "Avoid premium upselling — price sensitivity is high here.",
        ],
        "color": "#E63946",
    },
    1: {
        "title": "Cluster 1 — High-Value Premium Buyers",
        "insight": "High income, high spending across all channels. Fewer children, older age group. Active on web, store, and catalog.",
        "strategy": [
            "Offer premium product lines and curated catalog recommendations.",
            "Reward loyalty with early access to new products or exclusive offers.",
            "Use catalog marketing — this cluster responds to it.",
            "Focus on wine and meat categories — biggest spend areas.",
        ],
        "color": "#457B9D",
    },
    2: {
        "title": "Cluster 2 — Budget-Conscious Browsers",
        "insight": "Lowest income and spending. Most children at home. Highest web visits but lowest purchases — price-sensitive window shoppers.",
        "strategy": [
            "Focus on deal alerts, flash sales, and daily discount notifications.",
            "Promote essentials and low-ticket categories (fruits, sweets).",
            "Use email nudges for abandoned browsing sessions.",
            "Do not invest heavily in catalog or store promotions for this group.",
        ],
        "color": "#F4A261",
    },
    3: {
        "title": "Cluster 3 — High-Income Campaign Responders",
        "insight": "Similar income and spending to Cluster 1 but with a standout 30% campaign response rate. Fewest children. Highest ROI per campaign spend.",
        "strategy": [
            "Prioritise this cluster for all direct campaigns — highest conversion rate.",
            "Upsell premium and gold product categories.",
            "Build referral and loyalty programs — most likely to advocate.",
            "Test new products or offers here first before broader rollout.",
        ],
        "color": "#2A9D8F",
    },
}

sns.set_theme(style="whitegrid", font_scale=1.0)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.title("SmartCart")
    st.markdown("**Customer Segmentation System**")
    st.markdown("---")
    uploaded = st.file_uploader("Upload smartcart_customers.csv", type=["csv"])
    st.markdown("---")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4, step=1)
    st.markdown("---")
    section = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🔍 Exploratory Analysis",
            "📐 Dimensionality Reduction",
            "🎯 K Selection",
            "🗂️ Cluster Results",
            "📈 Business Insights",
            "🔮 Predict My Segment",
        ],
    )

# ── Load & pipeline ───────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(file, k):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("smartcart_customers.csv")

    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["Customer_Tenure_days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    df["Total_spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Education"] = df["Education"].replace({
        "Basic": "UnderGraduate", "2n Cycle": "UnderGraduate",
        "Graduation": "Graduate", "PhD": "PostGraduate", "Master": "PostGraduate",
    })
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Together", "Together": "Together",
        "Divorced": "Single", "Alone": "Single",
        "Widow": "Single", "Absurd": "Single",
        "YOLO": "Single", "Single": "Single",
    })

    raw_with_feats = df.copy()

    drop_cols = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds",
    ]
    df_cleaned = df.drop(columns=drop_cols)
    df_cleaned = df_cleaned[(df_cleaned["Age"] < 90) & (df_cleaned["Income"] < 600_000)].copy()

    ohe = OneHotEncoder(sparse_output=False)
    cat_cols = ["Education", "Living_With"]
    enc_arr = ohe.fit_transform(df_cleaned[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=ohe.get_feature_names_out(cat_cols), index=df_cleaned.index)
    df_num = df_cleaned.drop(columns=cat_cols)
    df_encoded = pd.concat([df_num, enc_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Agglomerative for labelling existing data
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(X_pca)
    df_cleaned["Cluster"] = labels

    # KMeans fitted on same PCA space — used for .predict() on new inputs
    km_pred = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_pred.fit(X_pca)

    # Map KMeans labels to Agglomerative labels by cluster centroid proximity
    # so predictions are consistent with what's shown in the charts
    from scipy.spatial.distance import cdist
    agg_centroids = np.array([X_pca[labels == i].mean(axis=0) for i in range(k)])
    km_centroids = km_pred.cluster_centers_
    dist_matrix = cdist(km_centroids, agg_centroids)
    km_to_agg = dist_matrix.argmin(axis=1)  # for each KMeans cluster, closest Agg cluster

    sil = silhouette_score(X_pca, labels)

    raw_with_feats = raw_with_feats[
        (raw_with_feats["Age"] < 90) & (raw_with_feats["Income"] < 600_000)
    ].copy()
    raw_with_feats = raw_with_feats.loc[df_cleaned.index]
    raw_with_feats["Cluster"] = labels

    return df_cleaned, raw_with_feats, X_pca, X_scaled, pca, labels, sil, scaler, ohe, df_encoded.columns.tolist(), km_pred, km_to_agg


df_cleaned, raw_feats, X_pca, X_scaled, pca, labels, sil_score, scaler, ohe, feature_cols, km_pred, km_to_agg = load_and_process(uploaded, n_clusters)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)


def predict_cluster(age, income, total_spending, total_children, recency,
                    num_deals, num_web, num_catalog, num_store, num_web_visits,
                    complain, response, tenure_days, education, living_with):
    """Transform user inputs through the same pipeline and return predicted cluster."""
    # One-hot encode categorical inputs
    cat_input = pd.DataFrame([[education, living_with]], columns=["Education", "Living_With"])
    enc_cat = ohe.transform(cat_input)
    enc_cat_df = pd.DataFrame(enc_cat, columns=ohe.get_feature_names_out(["Education", "Living_With"]))

    num_input = pd.DataFrame([[
        income, recency, num_deals, num_web, num_catalog, num_store,
        num_web_visits, complain, response, age, tenure_days,
        total_spending, total_children,
    ]], columns=[
        "Income", "Recency", "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
        "Complain", "Response", "Age", "Customer_Tenure_days",
        "Total_spending", "Total_Children",
    ])

    full_input = pd.concat([num_input.reset_index(drop=True),
                             enc_cat_df.reset_index(drop=True)], axis=1)

    # Align columns to match training feature order
    full_input = full_input.reindex(columns=feature_cols, fill_value=0)

    X_new_scaled = scaler.transform(full_input)
    X_new_pca = pca.transform(X_new_scaled)

    km_label = km_pred.predict(X_new_pca)[0]
    agg_label = km_to_agg[km_label]
    return int(agg_label)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if section == "📊 Overview":
    st.title("🛒 SmartCart Customer Segmentation")
    st.markdown(
        "Unsupervised ML pipeline that segments **2,240 SmartCart customers** into "
        "distinct behavioural groups to enable targeted marketing and reduce churn."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{len(df_cleaned):,}")
    c2.metric("Clusters Found", n_clusters)
    c3.metric("Silhouette Score", f"{sil_score:.4f}")
    c4.metric("Avg Income", f"₹{df_cleaned['Income'].mean():,.0f}")
    c5.metric("Avg Spending", f"₹{df_cleaned['Total_spending'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset at a Glance")
        info = {
            "Records": "2,240",
            "Features (raw)": "22",
            "Missing values": "24 (Income only)",
            "Outliers removed": "4 (Age>90 or Income>600k)",
            "Final records": f"{len(df_cleaned):,}",
            "Engineered features": "6 new features created",
            "Algorithm": "Agglomerative Clustering (Ward)",
            "PCA components": "3 (45% variance explained)",
        }
        st.table(pd.DataFrame(info.items(), columns=["Property", "Value"]))

    with col2:
        st.subheader("Cluster Size Distribution")
        counts = df_cleaned["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color=PALETTE[:n_clusters], edgecolor="white")
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(v), ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Customers")
        ax.set_title("Customers per Cluster")
        fig.tight_layout()
        fig_to_st(fig)

    st.markdown("---")
    st.subheader("Cluster Quick Reference")
    summary = df_cleaned.groupby("Cluster")[
        ["Income", "Total_spending", "Age", "Total_Children", "NumWebPurchases",
         "NumStorePurchases", "NumCatalogPurchases", "NumWebVisitsMonth", "Response"]
    ].mean().round(1)
    summary.index = [f"Cluster {i}" for i in summary.index]
    summary.columns = ["Avg Income", "Avg Spending", "Avg Age", "Avg Children",
                        "Web Purchases", "Store Purchases", "Catalog Purchases",
                        "Web Visits/mo", "Campaign Response"]
    st.dataframe(summary.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Exploratory Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Pairplot", "Heatmap", "Spending Breakdown"])

    with tab1:
        st.subheader("Feature Distributions")
        feat = st.selectbox("Select feature", ["Income", "Total_spending", "Age", "Recency",
                                                 "Total_Children", "Customer_Tenure_days",
                                                 "NumWebVisitsMonth"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df_cleaned[feat], kde=True, ax=axes[0], color="#457B9D")
        axes[0].set_title(f"{feat} — Distribution")
        sns.boxplot(y=df_cleaned[feat], ax=axes[1], color="#F4A261")
        axes[1].set_title(f"{feat} — Boxplot")
        fig.tight_layout()
        fig_to_st(fig)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{df_cleaned[feat].mean():,.1f}")
        col2.metric("Median", f"{df_cleaned[feat].median():,.1f}")
        col3.metric("Std Dev", f"{df_cleaned[feat].std():,.1f}")
        col4.metric("Range", f"{df_cleaned[feat].min():,.0f} – {df_cleaned[feat].max():,.0f}")

    with tab2:
        st.subheader("Pairplot — Key Features")
        st.caption("Shows pairwise relationships and KDE distributions across core numerical features.")
        cols_pp = ["Income", "Recency", "Age", "Total_spending", "Total_Children"]
        with st.spinner("Rendering pairplot..."):
            g = sns.pairplot(df_cleaned[cols_pp], diag_kind="kde",
                             plot_kws=dict(alpha=0.35, s=8), corner=True)
            g.fig.suptitle("Pairplot — Key Features", y=1.02, fontsize=13, fontweight="bold")
            fig_to_st(g.fig)

    with tab3:
        st.subheader("Correlation Heatmap")
        corr = df_cleaned.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)
        st.info("**Key insight:** Income and Total_spending are strongly correlated. "
                "Catalog purchases correlate more with high-income segments than web purchases do.")

    with tab4:
        st.subheader("Spending per Category")
        spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                      "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
        cat_labels = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]
        spend_means = raw_feats[spend_cols].mean()
        spend_means.index = cat_labels
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(cat_labels, spend_means.values, color=PALETTE * 2, edgecolor="white")
        axes[0].set_title("Overall Avg Spending per Category", fontweight="bold")
        axes[0].set_ylabel("Avg Amount Spent")
        cat_cluster = raw_feats.groupby("Cluster")[spend_cols].mean()
        cat_cluster.columns = cat_labels
        x = np.arange(len(cat_labels))
        w = 0.2
        for i, c in enumerate(PALETTE[:n_clusters]):
            axes[1].bar(x + i * w, cat_cluster.iloc[i], w, label=f"Cluster {i}", color=c)
        axes[1].set_xticks(x + w * (n_clusters - 1) / 2)
        axes[1].set_xticklabels(cat_labels)
        axes[1].set_title("Spending per Category by Cluster", fontweight="bold")
        axes[1].set_ylabel("Avg Amount Spent")
        axes[1].legend()
        fig.tight_layout()
        fig_to_st(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Dimensionality Reduction
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📐 Dimensionality Reduction":
    st.title("📐 PCA — Dimensionality Reduction")

    var = pca.explained_variance_ratio_
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PC1 Variance", f"{var[0]*100:.1f}%")
    c2.metric("PC2 Variance", f"{var[1]*100:.1f}%")
    c3.metric("PC3 Variance", f"{var[2]*100:.1f}%")
    c4.metric("Total Explained", f"{var.sum()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Explained Variance per Component")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["PC1", "PC2", "PC3"], var * 100, color=PALETTE[:3], edgecolor="white")
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title("PCA Explained Variance")
        for i, v in enumerate(var):
            ax.text(i, v * 100 + 0.3, f"{v*100:.1f}%", ha="center", fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

    with col2:
        st.subheader("3D Projection — Pre-Clustering")
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.35, s=8, color="#457B9D")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title("3D PCA Projection", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

    st.subheader("3D PCA — Coloured by Cluster")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, c in enumerate(PALETTE[:n_clusters]):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   alpha=0.6, s=10, color=c, label=f"Cluster {i}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Agglomerative Clusters in PCA Space", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig_to_st(fig)

    st.warning(
        "⚠️ **Note:** 3 PCA components explain only ~45% of total variance. "
        "Clusters are found in this reduced space — real-world separation may be "
        "stronger in the original 18-feature space."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — K Selection
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🎯 K Selection":
    st.title("🎯 Optimal K Selection")

    with st.spinner("Computing WCSS and Silhouette scores..."):
        wcss, sil_scores = [], []
        for k in range(1, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_pca)
            wcss.append(km.inertia_)
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_pca)
            sil_scores.append(silhouette_score(X_pca, lbl))
        knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Method (WCSS)")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, 11), wcss, marker="o", color="#457B9D", linewidth=2, markersize=7)
        ax.axvline(x=knee.elbow, color="#E63946", linestyle="--", linewidth=1.8,
                   label=f"Elbow at k={knee.elbow}")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("WCSS (Inertia)")
        ax.set_title("Elbow Method", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig_to_st(fig)
        st.info(f"KneeLocator identified elbow at **k = {knee.elbow}**")

    with col2:
        st.subheader("Silhouette Score")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(2, 11), sil_scores, marker="o", color="#2A9D8F", linewidth=2, markersize=7)
        ax.axvline(x=4, color="#E63946", linestyle="--", linewidth=1.8, label="Selected k=4")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score vs K", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig_to_st(fig)
        best_k = np.argmax(sil_scores) + 2
        st.info(f"Silhouette peaks at **k = {best_k}** (score: {max(sil_scores):.4f})")

    st.markdown("---")
    st.subheader("Score Table")
    score_df = pd.DataFrame({
        "k": range(2, 11),
        "Silhouette Score": [round(s, 4) for s in sil_scores],
    })
    st.dataframe(score_df, use_container_width=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Cluster Results
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🗂️ Cluster Results":
    st.title("🗂️ Cluster Results")

    tab1, tab2, tab3 = st.tabs(["Scatter Plots", "Radar Chart", "Detailed Profiles"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Income vs Total Spending")
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, c in enumerate(PALETTE[:n_clusters]):
                sub = df_cleaned[df_cleaned["Cluster"] == i]
                ax.scatter(sub["Total_spending"], sub["Income"], alpha=0.5, s=18,
                           color=c, label=f"Cluster {i}")
            ax.set_xlabel("Total Spending")
            ax.set_ylabel("Income")
            ax.set_title("Income vs Spending by Cluster", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            fig_to_st(fig)

        with col2:
            st.subheader("Age vs Total Spending")
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, c in enumerate(PALETTE[:n_clusters]):
                sub = df_cleaned[df_cleaned["Cluster"] == i]
                ax.scatter(sub["Age"], sub["Total_spending"], alpha=0.5, s=18,
                           color=c, label=f"Cluster {i}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Total Spending")
            ax.set_title("Age vs Spending by Cluster", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            fig_to_st(fig)

        st.subheader("Web Visits vs Purchases")
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, c in enumerate(PALETTE[:n_clusters]):
            sub = df_cleaned[df_cleaned["Cluster"] == i]
            ax.scatter(sub["NumWebVisitsMonth"], sub["NumWebPurchases"],
                       alpha=0.45, s=18, color=c, label=f"Cluster {i}")
        ax.set_xlabel("Web Visits per Month")
        ax.set_ylabel("Web Purchases")
        ax.set_title("Web Visits vs Web Purchases — conversion gap visible for Clusters 0 & 2",
                     fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig_to_st(fig)

    with tab2:
        st.subheader("Radar Chart — Normalised Cluster Profiles")
        metrics_r = ["Income", "Total_spending", "NumWebPurchases",
                     "NumStorePurchases", "NumCatalogPurchases", "NumWebVisitsMonth"]
        labels_r = ["Income", "Spending", "Web\nPurchases",
                    "Store\nPurchases", "Catalog\nPurchases", "Web\nVisits"]
        cm = df_cleaned.groupby("Cluster")[metrics_r].mean()
        cn = (cm - cm.min()) / (cm.max() - cm.min())
        N = len(metrics_r)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, c in enumerate(PALETTE[:n_clusters]):
            vals = cn.iloc[i].tolist() + [cn.iloc[i].tolist()[0]]
            ax.plot(angles, vals, color=c, linewidth=2, label=f"Cluster {i}")
            ax.fill(angles, vals, color=c, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_r, fontsize=11)
        ax.set_title("Cluster Profiles — Radar Chart", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        fig_to_st(fig)

    with tab3:
        st.subheader("Select a Cluster to Inspect")
        chosen = st.selectbox("Cluster", options=list(range(n_clusters)),
                               format_func=lambda x: f"Cluster {x} — {CLUSTER_NAMES.get(x, '')}")
        sub = df_cleaned[df_cleaned["Cluster"] == chosen]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", len(sub))
        c2.metric("Avg Income", f"₹{sub['Income'].mean():,.0f}")
        c3.metric("Avg Spending", f"₹{sub['Total_spending'].mean():,.0f}")
        c4.metric("Campaign Response", f"{sub['Response'].mean()*100:.0f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Behavioural Averages**")
            profile = sub[["Age", "Total_Children", "Recency", "NumWebPurchases",
                            "NumStorePurchases", "NumCatalogPurchases",
                            "NumDealsPurchases", "NumWebVisitsMonth"]].mean().round(1)
            st.dataframe(profile.rename("Mean Value"), use_container_width=True)

        with col2:
            st.markdown("**Spending Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(sub["Total_spending"], kde=True, color=PALETTE[chosen % len(PALETTE)], ax=ax)
            ax.set_xlabel("Total Spending")
            ax.set_title(f"Cluster {chosen} — Spending Distribution")
            fig.tight_layout()
            fig_to_st(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Business Insights
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 Business Insights":
    st.title("📈 Business Insights & Recommendations")

    cluster_cards = [
        {
            "id": 0,
            "label": "Cluster 0 — Mid-Income Moderate Spenders",
            "size": "905 customers (40%)",
            "income": "₹39,681",
            "spending": "₹222",
            "insight": "Largest group. Middle income, moderate spending, more children at home. "
                       "High web visits but low purchases — browsing without converting.",
            "strategy": "Conversion campaigns, discount nudges, family product bundles.",
            "color": "#E63946",
        },
        {
            "id": 1,
            "label": "Cluster 1 — High-Value Premium Buyers",
            "size": "534 customers (24%)",
            "income": "₹72,808",
            "spending": "₹1,237",
            "insight": "High income, high spending across all channels. Fewer children, older. "
                       "Active on web, store, and catalog.",
            "strategy": "Premium product recommendations, loyalty rewards, early access to new launches.",
            "color": "#457B9D",
        },
        {
            "id": 2,
            "label": "Cluster 2 — Budget-Conscious Browsers",
            "size": "444 customers (20%)",
            "income": "₹36,960",
            "spending": "₹166",
            "insight": "Lowest income and spending. Highest web visits but lowest purchase rate — "
                       "price-sensitive window shoppers. Most children at home.",
            "strategy": "Budget-friendly promotions, deal alerts, essentials category targeting.",
            "color": "#F4A261",
        },
        {
            "id": 3,
            "label": "Cluster 3 — High-Income Campaign Responders",
            "size": "353 customers (16%)",
            "income": "₹70,723",
            "spending": "₹1,190",
            "insight": "Similar to Cluster 1 in income and spending but standout campaign "
                       "response rate (30%). Fewest children. Highest conversion potential.",
            "strategy": "Direct campaign targeting, upselling premium categories, referral programs.",
            "color": "#2A9D8F",
        },
    ]

    for card in cluster_cards:
        with st.container():
            st.markdown(
                f"""
                <div style="border-left: 5px solid {card['color']}; padding: 12px 18px;
                            margin-bottom: 18px; background: #f8f9fa; border-radius: 4px;">
                    <h4 style="margin:0; color:{card['color']}">{card['label']}</h4>
                    <p style="margin:4px 0; color:#555; font-size:0.9rem">
                        <b>Size:</b> {card['size']} &nbsp;|&nbsp;
                        <b>Avg Income:</b> {card['income']} &nbsp;|&nbsp;
                        <b>Avg Spending:</b> {card['spending']}
                    </p>
                    <p style="margin:6px 0"><b>Observation:</b> {card['insight']}</p>
                    <p style="margin:4px 0"><b>Strategy:</b> {card['strategy']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Key Findings")
    findings = [
        ("Income drives spending", "High-income clusters (1, 3) spend 5–7× more than low-income clusters (0, 2)."),
        ("Children reduce spending", "Clusters with more children consistently spend less, across income levels."),
        ("High web visits ≠ high purchases", "Clusters 0 and 2 visit most but buy least — a conversion problem, not a reach problem."),
        ("Recency is similar across all clusters (~49 days)", "Engagement recency doesn't differentiate segments — spending behaviour does."),
        ("Catalog purchases signal premium buyers", "Near-zero for low-spend clusters, high for Clusters 1 and 3."),
        ("Deals work for low-income segments", "Clusters 0 and 2 use discounts more than high-income clusters."),
    ]
    for title, body in findings:
        st.markdown(f"- **{title}:** {body}")

    st.markdown("---")
    st.subheader("Limitations")
    lims = [
        "PCA at 3 components explains only 45% of variance — some structure is lost before clustering.",
        "Silhouette score of 0.38 is moderate — segments are real but not sharply defined.",
        "Recency doesn't differentiate clusters — cannot flag dormant users from cluster membership alone.",
        "Agglomerative Clustering doesn't scale to millions of records — production needs KMeans or HDBSCAN.",
    ]
    for l in lims:
        st.markdown(f"- {l}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Predict My Segment
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔮 Predict My Segment":
    st.title("🔮 Predict Customer Segment")
    st.markdown(
        "Enter a customer's details below and the model will predict which segment they belong to, "
        "along with the recommended marketing strategy for that segment."
    )
    st.info(
        "**How it works:** Your inputs are passed through the same scaler and PCA used during training. "
        "A KMeans model (fitted on the same PCA space) predicts the nearest cluster. "
        "Labels are mapped to match the Agglomerative clusters shown in the charts."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", min_value=18, max_value=85, value=40, step=1)
        income = st.number_input("Annual Income (₹)", min_value=5000, max_value=500000,
                                  value=50000, step=1000)
        education = st.selectbox("Education Level",
                                  options=["Graduate", "PostGraduate", "UnderGraduate"])
        living_with = st.selectbox("Living With", options=["Together", "Single"])
        total_children = st.slider("Total Children at Home", min_value=0, max_value=4, value=1)
        tenure_days = st.slider("Days Since Enrollment", min_value=0, max_value=3000, value=1000)

    with col2:
        st.subheader("Spending")
        total_spending = st.number_input("Total Spending (₹)", min_value=0, max_value=2600,
                                          value=500, step=10)
        st.caption("Rough total across wines, meat, fish, fruits, sweets, and gold.")
        recency = st.slider("Days Since Last Purchase", min_value=0, max_value=99, value=45)
        complain = st.selectbox("Complained in Last 2 Years", options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")
        response = st.selectbox("Responded to Last Campaign", options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        st.subheader("Purchase Behaviour")
        num_web = st.slider("Web Purchases (last period)", min_value=0, max_value=27, value=4)
        num_store = st.slider("Store Purchases", min_value=0, max_value=13, value=5)
        num_catalog = st.slider("Catalog Purchases", min_value=0, max_value=28, value=2)
        num_deals = st.slider("Discount / Deal Purchases", min_value=0, max_value=15, value=2)
        num_web_visits = st.slider("Web Visits per Month", min_value=0, max_value=20, value=5)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Segment", use_container_width=True, type="primary")

    if predict_btn:
        cluster_id = predict_cluster(
            age=age, income=income, total_spending=total_spending,
            total_children=total_children, recency=recency,
            num_deals=num_deals, num_web=num_web, num_catalog=num_catalog,
            num_store=num_store, num_web_visits=num_web_visits,
            complain=complain, response=response,
            tenure_days=tenure_days, education=education, living_with=living_with,
        )

        info = CLUSTER_STRATEGIES.get(cluster_id, {
            "title": f"Cluster {cluster_id}",
            "insight": "Custom cluster from adjusted k setting.",
            "strategy": ["Apply targeted marketing based on cluster characteristics."],
            "color": PALETTE[cluster_id % len(PALETTE)],
        })

        st.markdown("### Prediction Result")
        st.markdown(
            f"""
            <div style="border-left: 6px solid {info['color']}; padding: 16px 20px;
                        background: #f8f9fa; border-radius: 6px; margin-bottom: 20px;">
                <h3 style="margin:0; color:{info['color']}">Cluster {cluster_id} — {CLUSTER_NAMES.get(cluster_id, 'Custom Cluster')}</h3>
                <p style="margin-top: 10px; font-size: 1rem">{info['insight']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Recommended Marketing Strategy")
        for point in info["strategy"]:
            st.markdown(f"- {point}")

        st.markdown("---")
        st.markdown("#### Where This Customer Sits vs Cluster Averages")

        avg = df_cleaned[df_cleaned["Cluster"] == cluster_id][
            ["Income", "Total_spending", "Age", "Total_Children",
             "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth"]
        ].mean().round(1)

        user_vals = pd.Series({
            "Income": income,
            "Total_spending": total_spending,
            "Age": age,
            "Total_Children": total_children,
            "NumWebPurchases": num_web,
            "NumStorePurchases": num_store,
            "NumWebVisitsMonth": num_web_visits,
        })

        compare_df = pd.DataFrame({
            "Your Input": user_vals,
            f"Cluster {cluster_id} Avg": avg,
        })
        st.dataframe(compare_df.style.background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)

        # Mini bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        metrics_show = ["Income", "Total_spending", "NumWebPurchases", "NumStorePurchases"]
        labels_show = ["Income", "Spending", "Web Purchases", "Store Purchases"]
        x = np.arange(len(metrics_show))
        w = 0.35
        axes[0].bar(x - w/2, [user_vals[m] for m in metrics_show], w,
                    label="Your Input", color=info["color"], alpha=0.85)
        axes[0].bar(x + w/2, [avg[m] for m in metrics_show], w,
                    label=f"Cluster {cluster_id} Avg", color="#AAAAAA", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels_show)
        axes[0].set_title("Your Profile vs Cluster Average", fontweight="bold")
        axes[0].legend()

        # Cluster distribution with user marker on income vs spending
        for i, c in enumerate(PALETTE[:n_clusters]):
            sub = df_cleaned[df_cleaned["Cluster"] == i]
            axes[1].scatter(sub["Total_spending"], sub["Income"],
                            alpha=0.25, s=12, color=c, label=f"Cluster {i}")
        axes[1].scatter(total_spending, income, color="black", s=180,
                        zorder=5, marker="*", label="You")
        axes[1].set_xlabel("Total Spending")
        axes[1].set_ylabel("Income")
        axes[1].set_title("Your Position in the Dataset", fontweight="bold")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig_to_st(fig)
