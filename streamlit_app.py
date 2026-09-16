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
from sklearn.neighbors import NearestCentroid
from kneed import KneeLocator
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartCart Segmentation", page_icon="🛒", layout="wide")

PALETTE = ["#E63946", "#457B9D", "#F4A261", "#2A9D8F"]
CLUSTER_NAMES = {
    0: "Mid-Income Moderate Spenders",
    1: "High-Value Premium Buyers",
    2: "Budget-Conscious Browsers",
    3: "High-Income Campaign Responders",
}
STRATEGIES = {
    0: ["Run discount-led conversion campaigns.", "Promote family product bundles.", "Retarget frequent browsers who don't buy."],
    1: ["Offer premium product lines and catalog recommendations.", "Reward loyalty with early access.", "Focus on wine and meat categories."],
    2: ["Push deal alerts and flash sales.", "Promote low-ticket categories (fruits, sweets).", "Don't invest heavily in catalog or store promotions."],
    3: ["Prioritise for all direct campaigns — 30% response rate.", "Upsell premium and gold categories.", "Build referral and loyalty programs."],
}

# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_data
def run_pipeline():
    df = pd.read_csv("smartcart_customers.csv")
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["Customer_Tenure_days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    df["Total_spending"] = df[["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Education"] = df["Education"].replace({"Basic":"UnderGraduate","2n Cycle":"UnderGraduate","Graduation":"Graduate","PhD":"PostGraduate","Master":"PostGraduate"})
    df["Living_With"] = df["Marital_Status"].replace({"Married":"Together","Together":"Together","Divorced":"Single","Alone":"Single","Widow":"Single","Absurd":"Single","YOLO":"Single","Single":"Single"})

    raw = df.copy()
    drop_cols = ["ID","Year_Birth","Marital_Status","Kidhome","Teenhome","Dt_Customer","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    df = df.drop(columns=drop_cols)
    df = df[(df["Age"] < 90) & (df["Income"] < 600_000)].copy()

    ohe = OneHotEncoder(sparse_output=False)
    enc = ohe.fit_transform(df[["Education","Living_With"]])
    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(["Education","Living_With"]), index=df.index)
    df_enc = pd.concat([df.drop(columns=["Education","Living_With"]), enc_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_enc)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels = agg.fit_predict(X_pca)
    df["Cluster"] = labels

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_pca)
    agg_centroids = np.array([X_pca[labels == i].mean(axis=0) for i in range(4)])
    km_to_agg = cdist(km.cluster_centers_, agg_centroids).argmin(axis=1)

    sil = silhouette_score(X_pca, labels)

    raw = raw[(raw["Age"] < 90) & (raw["Income"] < 600_000)].copy()
    raw = raw.loc[df.index]
    raw["Cluster"] = labels

    return df, raw, X_pca, pca, scaler, ohe, df_enc.columns.tolist(), km, km_to_agg, labels, sil

df, raw, X_pca, pca, scaler, ohe, feat_cols, km, km_to_agg, labels, sil = run_pipeline()

def predict(age, income, spending, children, recency, deals, web, catalog, store, visits, complain, response, tenure, education, living_with):
    cat = pd.DataFrame([[education, living_with]], columns=["Education","Living_With"])
    enc = pd.DataFrame(ohe.transform(cat), columns=ohe.get_feature_names_out(["Education","Living_With"]))
    num = pd.DataFrame([[income, recency, deals, web, catalog, store, visits, complain, response, age, tenure, spending, children]],
                       columns=["Income","Recency","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth","Complain","Response","Age","Customer_Tenure_days","Total_spending","Total_Children"])
    row = pd.concat([num, enc], axis=1).reindex(columns=feat_cols, fill_value=0)
    X_new = pca.transform(scaler.transform(row))
    return int(km_to_agg[km.predict(X_new)[0]])

def close(fig): st.pyplot(fig); plt.close(fig)

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 SmartCart")
    st.caption("Customer Segmentation System")
    st.markdown("---")
    section = st.radio("Navigate", ["📊 Overview","🔍 EDA","📐 PCA","🎯 K Selection","🗂️ Clusters","📈 Insights","🔮 Predict"])

# ══════════════════════════════════════════════════════════════════════════════
if section == "📊 Overview":
    st.title("SmartCart Customer Segmentation")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Clusters", 4)
    c3.metric("Silhouette", f"{sil:.4f}")
    c4.metric("Avg Income", f"₹{df['Income'].mean():,.0f}")
    c5.metric("Avg Spending", f"₹{df['Total_spending'].mean():,.0f}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Summary")
        st.table(pd.DataFrame({"Property":["Raw records","Features","Missing values","Outliers removed","Final records","Algorithm","PCA components"],
                                "Value":["2,240","22","24 (Income)","4","2,236","Agglomerative (Ward)","3 (~45% variance)"]}).set_index("Property"))
    with col2:
        st.subheader("Cluster Sizes")
        counts = df["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white")
        for b,v in zip(bars, counts.values): ax.text(b.get_x()+b.get_width()/2, b.get_height()+5, str(v), ha="center", fontweight="bold")
        ax.set_xlabel("Cluster"); ax.set_ylabel("Count"); fig.tight_layout(); close(fig)

    st.markdown("---")
    st.subheader("Cluster Summary Table")
    s = df.groupby("Cluster")[["Income","Total_spending","Age","Total_Children","NumWebPurchases","NumStorePurchases","NumWebVisitsMonth","Response"]].mean().round(1)
    s.index = [f"Cluster {i}" for i in s.index]
    s.columns = ["Avg Income","Avg Spending","Avg Age","Children","Web Purchases","Store Purchases","Web Visits/mo","Response Rate"]
    st.dataframe(s.style.background_gradient(cmap="Blues"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔍 EDA":
    st.title("Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlation Heatmap", "Spending by Category"])

    with tab1:
        feat = st.selectbox("Feature", ["Income","Total_spending","Age","Recency","Total_Children","NumWebVisitsMonth"])
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        sns.histplot(df[feat], kde=True, ax=axes[0], color="#457B9D"); axes[0].set_title(f"{feat} — Distribution")
        sns.boxplot(y=df[feat], ax=axes[1], color="#F4A261"); axes[1].set_title(f"{feat} — Boxplot")
        fig.tight_layout(); close(fig)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean", f"{df[feat].mean():,.1f}"); c2.metric("Median", f"{df[feat].median():,.1f}")
        c3.metric("Std Dev", f"{df[feat].std():,.1f}"); c4.metric("Range", f"{df[feat].min():,.0f}–{df[feat].max():,.0f}")

    with tab2:
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(11,9))
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr,bool)), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5, annot_kws={"size":8})
        ax.set_title("Correlation Heatmap", fontweight="bold"); fig.tight_layout(); close(fig)

    with tab3:
        spend_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
        cat_labels = ["Wines","Fruits","Meat","Fish","Sweets","Gold"]
        cluster_spend = raw.groupby("Cluster")[spend_cols].mean()
        cluster_spend.columns = cat_labels
        x = np.arange(len(cat_labels)); w = 0.2
        fig, ax = plt.subplots(figsize=(11,5))
        for i,c in enumerate(PALETTE): ax.bar(x+i*w, cluster_spend.iloc[i], w, label=f"Cluster {i}", color=c)
        ax.set_xticks(x+w*1.5); ax.set_xticklabels(cat_labels); ax.set_ylabel("Avg Spend")
        ax.set_title("Spending per Category by Cluster", fontweight="bold"); ax.legend(); fig.tight_layout(); close(fig)

# ══════════════════════════════════════════════════════════════════════════════
elif section == "📐 PCA":
    st.title("PCA — Dimensionality Reduction")
    var = pca.explained_variance_ratio_
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("PC1",f"{var[0]*100:.1f}%"); c2.metric("PC2",f"{var[1]*100:.1f}%")
    c3.metric("PC3",f"{var[2]*100:.1f}%"); c4.metric("Total",f"{var.sum()*100:.1f}%")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(["PC1","PC2","PC3"], var*100, color=PALETTE[:3], edgecolor="white")
        for i,v in enumerate(var): ax.text(i, v*100+0.3, f"{v*100:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Variance Explained (%)"); ax.set_title("PCA Variance"); fig.tight_layout(); close(fig)
    with col2:
        fig = plt.figure(figsize=(5,4)); ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], alpha=0.3, s=6, color="#457B9D")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title("3D PCA (pre-clustering)"); fig.tight_layout(); close(fig)

    fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111, projection="3d")
    for i,c in enumerate(PALETTE):
        m = labels==i; ax.scatter(X_pca[m,0], X_pca[m,1], X_pca[m,2], alpha=0.6, s=8, color=c, label=f"Cluster {i}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("3D PCA — Agglomerative Clusters", fontweight="bold"); ax.legend(); fig.tight_layout(); close(fig)
    st.warning("3 components explain ~45% of variance. Clusters are approximations in this reduced space.")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🎯 K Selection":
    st.title("Optimal K Selection")
    with st.spinner("Computing scores..."):
        wcss, sils = [], []
        for k in range(1,11):
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10); km_tmp.fit(X_pca); wcss.append(km_tmp.inertia_)
        for k in range(2,11):
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10); sils.append(silhouette_score(X_pca, km_tmp.fit_predict(X_pca)))
        knee = KneeLocator(range(1,11), wcss, curve="convex", direction="decreasing")

    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1,11), wcss, marker="o", color="#457B9D", linewidth=2)
        ax.axvline(x=knee.elbow, color="#E63946", linestyle="--", label=f"Elbow k={knee.elbow}")
        ax.set_xlabel("k"); ax.set_ylabel("WCSS"); ax.set_title("Elbow Method", fontweight="bold"); ax.legend(); fig.tight_layout(); close(fig)
        st.info(f"Elbow at **k = {knee.elbow}**")
    with col2:
        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(range(2,11), sils, marker="o", color="#2A9D8F", linewidth=2)
        ax.axvline(x=4, color="#E63946", linestyle="--", label="Selected k=4")
        ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score"); ax.set_title("Silhouette vs K", fontweight="bold"); ax.legend(); fig.tight_layout(); close(fig)
        st.info(f"Silhouette peaks at **k = {np.argmax(sils)+2}** ({max(sils):.4f})")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🗂️ Clusters":
    st.title("Cluster Results")
    tab1,tab2,tab3 = st.tabs(["Scatter Plots","Radar Chart","Per-Cluster Detail"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4))
            for i,c in enumerate(PALETTE):
                s = df[df["Cluster"]==i]; ax.scatter(s["Total_spending"], s["Income"], alpha=0.4, s=14, color=c, label=f"Cluster {i}")
            ax.set_xlabel("Spending"); ax.set_ylabel("Income"); ax.set_title("Income vs Spending", fontweight="bold"); ax.legend(); fig.tight_layout(); close(fig)
        with col2:
            fig,ax = plt.subplots(figsize=(6,4))
            for i,c in enumerate(PALETTE):
                s = df[df["Cluster"]==i]; ax.scatter(s["NumWebVisitsMonth"], s["NumWebPurchases"], alpha=0.4, s=14, color=c, label=f"Cluster {i}")
            ax.set_xlabel("Web Visits/mo"); ax.set_ylabel("Web Purchases"); ax.set_title("Web Visits vs Purchases", fontweight="bold"); ax.legend(); fig.tight_layout(); close(fig)

    with tab2:
        metrics_r = ["Income","Total_spending","NumWebPurchases","NumStorePurchases","NumCatalogPurchases","NumWebVisitsMonth"]
        labels_r  = ["Income","Spending","Web\nPurchases","Store\nPurchases","Catalog\nPurchases","Web\nVisits"]
        cm = df.groupby("Cluster")[metrics_r].mean()
        cn = (cm-cm.min())/(cm.max()-cm.min())
        N = len(metrics_r); angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
        fig,ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
        for i,c in enumerate(PALETTE):
            v = cn.iloc[i].tolist()+[cn.iloc[i].tolist()[0]]
            ax.plot(angles,v,color=c,linewidth=2,label=f"Cluster {i}"); ax.fill(angles,v,color=c,alpha=0.1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_r)
        ax.set_title("Cluster Profiles — Radar", fontweight="bold", pad=20); ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1)); fig.tight_layout(); close(fig)

    with tab3:
        chosen = st.selectbox("Cluster", range(4), format_func=lambda x: f"Cluster {x} — {CLUSTER_NAMES[x]}")
        sub = df[df["Cluster"]==chosen]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Customers", len(sub)); c2.metric("Avg Income", f"₹{sub['Income'].mean():,.0f}")
        c3.metric("Avg Spending", f"₹{sub['Total_spending'].mean():,.0f}"); c4.metric("Response Rate", f"{sub['Response'].mean()*100:.0f}%")
        col1,col2 = st.columns(2)
        with col1:
            st.dataframe(sub[["Age","Total_Children","Recency","NumWebPurchases","NumStorePurchases","NumDealsPurchases","NumWebVisitsMonth"]].mean().round(1).rename("Cluster Avg"), use_container_width=True)
        with col2:
            fig,ax = plt.subplots(figsize=(5,4))
            sns.histplot(sub["Total_spending"], kde=True, color=PALETTE[chosen], ax=ax)
            ax.set_xlabel("Total Spending"); ax.set_title("Spending Distribution"); fig.tight_layout(); close(fig)

# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 Insights":
    st.title("Business Insights & Recommendations")
    for cid, (name, color, size, income, spend, insight, strat) in {
        0: (CLUSTER_NAMES[0],"#E63946","905 (40%)","₹39,681","₹222","Largest group. Middle income, more children. High web visits but low purchases — browsers, not buyers.","Discount nudges, family bundles, retargeting."),
        1: (CLUSTER_NAMES[1],"#457B9D","534 (24%)","₹72,808","₹1,237","High income and spending across all channels. Fewer children, active on web/store/catalog.","Premium products, loyalty rewards, catalog marketing."),
        2: (CLUSTER_NAMES[2],"#F4A261","444 (20%)","₹36,960","₹166","Lowest income and spending. Most children. Highest web visits, lowest purchases — price sensitive.","Flash sales, deal alerts, essentials targeting."),
        3: (CLUSTER_NAMES[3],"#2A9D8F","353 (16%)","₹70,723","₹1,190","Similar to Cluster 1 but 30% campaign response rate. Fewest children. Highest ROI per campaign.","Prioritise for all campaigns, upsell premium, referral programs."),
    }.items():
        st.markdown(f"""<div style="border-left:5px solid {color};padding:12px 18px;margin-bottom:16px;background:#f8f9fa;border-radius:4px">
            <h4 style="margin:0;color:{color}">Cluster {cid} — {name}</h4>
            <p style="margin:4px 0;color:#555;font-size:0.88rem"><b>Size:</b> {size} &nbsp;|&nbsp; <b>Income:</b> {income} &nbsp;|&nbsp; <b>Spending:</b> {spend}</p>
            <p style="margin:6px 0">{insight}</p><p style="margin:4px 0"><b>Strategy:</b> {strat}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Key Findings")
    for t,b in [("Income drives spending","High-income clusters spend 5–7× more than low-income ones."),
                ("Children reduce spending","More children = less spending, across all income levels."),
                ("High web visits ≠ purchases","Clusters 0 & 2 visit most but buy least — conversion problem, not reach."),
                ("Recency doesn't differentiate","All clusters average ~49 days — spending behaviour is the real separator."),
                ("Catalog = premium signal","Near-zero for Clusters 0 & 2, high for 1 & 3."),
                ("Deals work for low-income","Clusters 0 & 2 use discounts more than high-income clusters.")]:
        st.markdown(f"- **{t}:** {b}")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔮 Predict":
    st.title("🔮 Predict Customer Segment")
    st.markdown("Enter customer details and get the segment they belong to with a tailored marketing recommendation.")
    st.markdown("---")

    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("Demographics")
        age          = st.slider("Age", 18, 85, 40)
        income       = st.number_input("Annual Income (₹)", 5000, 500000, 50000, step=1000)
        education    = st.selectbox("Education", ["Graduate","PostGraduate","UnderGraduate"])
        living_with  = st.selectbox("Living With", ["Together","Single"])
        children     = st.slider("Children at Home", 0, 4, 1)
        tenure       = st.slider("Days Since Enrollment", 0, 3000, 1000)

    with col2:
        st.subheader("Spending & Activity")
        spending     = st.number_input("Total Spending (₹)", 0, 2600, 400, step=10)
        recency      = st.slider("Days Since Last Purchase", 0, 99, 45)
        web_visits   = st.slider("Web Visits per Month", 0, 20, 5)
        complain     = st.selectbox("Complained Recently", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        response     = st.selectbox("Responded to Last Campaign", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    with col3:
        st.subheader("Purchase Channels")
        num_web      = st.slider("Web Purchases", 0, 27, 4)
        num_store    = st.slider("Store Purchases", 0, 13, 5)
        num_catalog  = st.slider("Catalog Purchases", 0, 28, 2)
        num_deals    = st.slider("Deal / Discount Purchases", 0, 15, 2)

    st.markdown("---")
    if st.button("🔍 Predict Segment", use_container_width=True, type="primary"):
        cid = predict(age, income, spending, children, recency, num_deals, num_web, num_catalog, num_store, web_visits, complain, response, tenure, education, living_with)
        color = PALETTE[cid]

        st.markdown(f"""<div style="border-left:6px solid {color};padding:16px 20px;background:#f8f9fa;border-radius:6px;margin-bottom:20px">
            <h3 style="margin:0;color:{color}">Cluster {cid} — {CLUSTER_NAMES[cid]}</h3>
        </div>""", unsafe_allow_html=True)

        st.subheader("Recommended Strategy")
        for point in STRATEGIES[cid]: st.markdown(f"- {point}")

        st.markdown("---")
        st.subheader("Your Profile vs Cluster Average")
        avg = df[df["Cluster"]==cid][["Income","Total_spending","Age","Total_Children","NumWebPurchases","NumStorePurchases","NumWebVisitsMonth"]].mean().round(1)
        user_vals = pd.Series({"Income":income,"Total_spending":spending,"Age":age,"Total_Children":children,"NumWebPurchases":num_web,"NumStorePurchases":num_store,"NumWebVisitsMonth":web_visits})
        st.dataframe(pd.DataFrame({"Your Input":user_vals, f"Cluster {cid} Avg":avg}), use_container_width=True)

        col1,col2 = st.columns(2)
        with col1:
            metrics_show = ["Income","Total_spending","NumWebPurchases","NumStorePurchases"]
            x = np.arange(len(metrics_show)); w = 0.35
            fig,ax = plt.subplots(figsize=(6,4))
            ax.bar(x-w/2, [user_vals[m] for m in metrics_show], w, label="You", color=color, alpha=0.85)
            ax.bar(x+w/2, [avg[m] for m in metrics_show], w, label=f"Cluster {cid} Avg", color="#AAAAAA", alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(["Income","Spending","Web Purch","Store Purch"], fontsize=9)
            ax.set_title("Your Profile vs Cluster Avg", fontweight="bold"); ax.legend(); fig.tight_layout(); close(fig)
        with col2:
            fig,ax = plt.subplots(figsize=(6,4))
            for i,c in enumerate(PALETTE):
                s = df[df["Cluster"]==i]; ax.scatter(s["Total_spending"], s["Income"], alpha=0.2, s=10, color=c, label=f"Cluster {i}")
            ax.scatter(spending, income, color="black", s=200, zorder=5, marker="*", label="You")
            ax.set_xlabel("Spending"); ax.set_ylabel("Income"); ax.set_title("Your Position", fontweight="bold"); ax.legend(fontsize=8); fig.tight_layout(); close(fig)
