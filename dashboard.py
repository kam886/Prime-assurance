"""
AssurPrime Dashboard — Crédit Agricole Assurances / Pacifica
Challenge Data Science – Multirisque Agricole Incendie
Run : streamlit run assurprime_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AssurPrime",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }

[data-testid="stSidebar"] { background:#0a1628; border-right:1px solid #162842; }
[data-testid="stSidebar"] * { color:#c8d8ea !important; }
[data-testid="stSidebar"] hr { border-color:#1e3a5f !important; }

[data-testid="stMetricValue"] { font-size:1.5rem !important; font-weight:700 !important; color:#0a1628 !important; }
[data-testid="stMetricLabel"] { font-size:0.75rem !important; color:#5a7a9a !important; text-transform:uppercase; letter-spacing:.5px; }
[data-testid="metric-container"] {
    background:#f5f8fc; border:1px solid #dde8f2;
    border-radius:8px; padding:16px 20px;
    border-top:3px solid #1565c0;
}
[data-testid="stMetricDelta"] { font-size:0.78rem !important; }

.block-container { padding-top:2rem; padding-bottom:1.5rem; }
h2 { color:#0a1628; font-weight:700; letter-spacing:-.3px; }
h3 { color:#1a3a5c; font-weight:600; }
h4 { color:#1565c0; font-weight:600; font-size:1.05rem; }

.stAlert > div { border-radius:8px; }
.badge-green  { background:#e8f5e9; color:#2e7d32; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.badge-yellow { background:#fff8e1; color:#f57f17; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.badge-red    { background:#ffebee; color:#c62828; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
BLUE   = "#1565c0"
ORANGE = "#e65100"
GREEN  = "#2e7d32"
RED    = "#c62828"
AMBER  = "#f57f17"
TEAL   = "#00695c"
GREY   = "#546e7a"

PALETTE = [BLUE, ORANGE, GREEN, RED, AMBER, TEAL, "#6a1b9a", GREY, "#00838f"]

THEME = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(245,248,252,.5)",
    font=dict(family="IBM Plex Sans,Arial,sans-serif", size=11, color="#1a2a3a"),
    margin=dict(t=50, b=30, l=20, r=20),
)

ACT_LABELS = {
    "ACT1": "Cultivateurs",
    "ACT2": "Éleveurs",
    "ACT3": "Viticulteurs",
    "ACT4": "Horticulteurs",
    "ACT5": "Polyculteurs",
    "ACT6": "Maraîchers",
    "ACT7": "Arboriculteurs",
    "ACT8": "Sylviculteurs",
    "ACT9": "Autres",
}

VOC_LABELS = {
    "VOC1": "Bovins",
    "VOC2": "Ovins/Caprins",
    "VOC3": "Porcins",
    "VOC4": "Volailles",
    "VOC5": "Lapins",
    "VOC6": "Grandes cultures",
    "VOC7": "Spécialisé",
    "VOC8": "Divers",
}

RMSE_FOLDS = {
    "Fold 1": 7043.98,
    "Fold 2": 6983.48,
    "Fold 3": 7170.57,
    "Fold 4": 6330.88,
    "Fold 5": 6440.77,
}


def fmt_eur(v, dec=0):
    return f"{v:,.{dec}f} €"


def sp_color(v):
    if v <= 70:
        return GREEN
    if v <= 100:
        return AMBER
    return RED


def sp_badge(v):
    if v <= 70:
        return "badge-green", "Rentable"
    if v <= 100:
        return "badge-yellow", "Limite"
    return "badge-red", "Déficitaire"


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    import os
    base = "/mnt/project" if os.path.exists("/mnt/project/x_train.csv") else "."

    x_tr = pd.read_csv(f"{base}/x_train.csv",   nrows=10_000, low_memory=False)
    y_tr = pd.read_csv(f"{base}/y_train.csv",   nrows=10_000)
    x_te = pd.read_csv(f"{base}/x_test.csv",    nrows=10_000, low_memory=False)
    sub  = pd.read_csv(f"{base}/submission.csv", nrows=10_000)

    # ── Train ──
    tr = x_tr.merge(y_tr[["ID", "FREQ", "CM", "CHARGE"]], on="ID", how="left")
    for c in ["FREQ", "CM", "CHARGE"]:
        tr[c] = tr[c].fillna(0.0)
    tr["SINISTRE"]   = (tr["CHARGE"] > 0).astype(int)
    tr["LOG_CHARGE"] = np.log1p(tr["CHARGE"])
    tr["ACT_LABEL"]  = tr["ACTIVIT2"].map(ACT_LABELS).fillna(tr["ACTIVIT2"])
    tr["VOC_LABEL"]  = tr["VOCATION"].map(VOC_LABELS).fillna(tr["VOCATION"])
    tr["ANCIE_BIN"]  = pd.cut(
        tr["ANCIENNETE"],
        bins=[-1, 0, 2, 5, 10, 999],
        labels=["0 an", "1-2 ans", "3-5 ans", "6-10 ans", "10+ ans"],
    ).astype(str)

    # ── Test / Prédictions ──
    te = x_te.merge(
        sub[["ID", "FREQ", "CM", "CHARGE"]].rename(columns={
            "FREQ":   "FREQ_PRED",
            "CM":     "CM_PRED",
            "CHARGE": "CHARGE_PRED",
        }),
        on="ID", how="left",
    )
    for c in ["FREQ_PRED", "CM_PRED", "CHARGE_PRED"]:
        te[c] = te[c].fillna(0.0)
    te["ACT_LABEL"] = te["ACTIVIT2"].map(ACT_LABELS).fillna(te["ACTIVIT2"])
    te["ANCIE_BIN"] = pd.cut(
        te["ANCIENNETE"],
        bins=[-1, 0, 2, 5, 10, 999],
        labels=["0 an", "1-2 ans", "3-5 ans", "6-10 ans", "10+ ans"],
    ).astype(str)

    return tr, te


with st.spinner("Chargement des données…"):
    train, test = load_data()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR — Navigation + Filtres
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 8px 6px'>
      <div style='font-size:2rem'>🌾</div>
      <div style='font-size:1.05rem;font-weight:700;color:#7dc3f5;letter-spacing:1.5px'>ASSURPRIME</div>
      <div style='font-size:0.7rem;color:#4a7fa0;margin-top:3px'>Modélisation du risque agricole</div>
      <div style='font-size:0.65rem;color:#3a5f7a;margin-top:1px'>Multirisque Agricole – Incendie</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Vue globale",
            "Portefeuille",
            "Sinistralité",
            "Écarts modèle & performance",
            "Rentabilité technique",
        "Modélisation & projections",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.82rem;font-weight:600;color:#7dc3f5;"
        "letter-spacing:.5px;margin-bottom:8px'>FILTRES</div>",
        unsafe_allow_html=True,
    )

    # Filtre activité — noms réels
    all_act_codes = sorted(train["ACTIVIT2"].dropna().unique().tolist())
    act_opts = ["Toutes"] + [
        f"{c} — {ACT_LABELS.get(c, c)}" for c in all_act_codes
    ]
    sel_act_raw = st.selectbox(
        "Activité agricole", act_opts,
        help="Principal facteur tarifaire en assurance MRA",
    )
    sel_act = "Toutes" if sel_act_raw == "Toutes" else sel_act_raw.split(" — ")[0]

    # Filtre zone géographique
    all_zone_vals = sorted(int(z) for z in train["ZONE"].dropna().unique())
    zone_opts = ["Toutes"] + [f"Zone {z}" for z in all_zone_vals]
    sel_zone_raw = st.selectbox(
        "Zone géographique", zone_opts,
        help="Principal facteur d'exposition climatique/géographique",
    )
    sel_zone = "Toutes" if sel_zone_raw == "Toutes" else sel_zone_raw.replace("Zone ", "")

    st.divider()
    n_tr = len(train)
    n_te = len(test)
    st.markdown(
        f"<span style='color:#5a8faa;font-size:0.73rem'>"
        f"Train : <b style='color:#a0c8e8'>{n_tr:,}</b> contrats<br>"
        f"Test  : <b style='color:#a0c8e8'>{n_te:,}</b> contrats<br>"
        f"Fenêtre : 10 000 premiers</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# FILTRE HELPER
# ─────────────────────────────────────────────────────────────────
def apply_filters(df, act, zone):
    m = pd.Series(True, index=df.index)
    if act != "Toutes":
        m &= df["ACTIVIT2"] == act
    if zone != "Toutes":
        m &= df["ZONE"].fillna(-1).astype(int).astype(str) == zone
    return df[m].copy()


df   = apply_filters(train, sel_act, sel_zone)
df_t = apply_filters(test,  sel_act, sel_zone)

if len(df) == 0:
    st.warning("Aucun contrat pour ces filtres. Modifiez la sélection.")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# HELPERS COMMUNS
# ─────────────────────────────────────────────────────────────────
def get_sp_data(tr_df, te_df, group_col):
    tr_g = tr_df.groupby(group_col).agg(
        sinistres_tot=("CHARGE", "sum"),
        sinistres_moy=("CHARGE", "mean"),
        freq_reel=("FREQ", "mean"),
        taux_sin=("SINISTRE", "mean"),
        n_train=("ID", "count"),
    ).reset_index()
    te_g = te_df.groupby(group_col).agg(
        prime_pred_moy=("CHARGE_PRED", "mean"),
        prime_pred_tot=("CHARGE_PRED", "sum"),
        freq_pred=("FREQ_PRED", "mean"),
        cm_pred=("CM_PRED", "mean"),
        n_test=("ID", "count"),
    ).reset_index()
    cmp = tr_g.merge(te_g, on=group_col, how="outer").fillna(0)
    cmp["sp_pct"]    = (cmp["sinistres_moy"] / (cmp["prime_pred_moy"] + 1e-9) * 100).round(1)
    cmp["ecart_moy"] = (cmp["sinistres_moy"] - cmp["prime_pred_moy"]).round(2)
    cmp["ecart_freq"]= (cmp["freq_reel"] - cmp["freq_pred"]).round(6)
    return cmp


def page_title(title, sub):
    st.markdown(f"## {title}")
    st.caption(sub)
    st.divider()


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 – VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════
if page == "Vue globale":
    page_title(
        "Synthèse globale",
        "Analyse globale du risque Multirisque Agricole Incendie",
    )

    n_c     = len(df)
    tot_sin = df["CHARGE"].sum()
    moy_sin = df["CHARGE"].mean()
    tx_sin  = df["SINISTRE"].mean() * 100
    n_sin   = int(df["SINISTRE"].sum())
    prime_p = df_t["CHARGE_PRED"].mean() if len(df_t) > 0 else 0
    sp_glob = (moy_sin / (prime_p + 1e-9) * 100) if prime_p > 0 else 0
    ann_m   = df["ANNEE_ASSURANCE"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Contrats",             f"{n_c:,}")
    k2.metric("Sinistres totaux",     fmt_eur(tot_sin))
    k3.metric("Charge moy./contrat",  f"{moy_sin:.1f} €")
    k4.metric("Taux de sinistralité", f"{tx_sin:.2f} %",
              delta=f"{n_sin} sinistres")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Prime pure prédite moy.", f"{prime_p:.1f} €")
    k6.metric("S/P ratio global",        f"{sp_glob:.1f} %",
              delta="Rentable" if sp_glob < 100 else "Déficitaire",
              delta_color="normal" if sp_glob < 100 else "inverse")
    k7.metric("Exposition moyenne",      f"{ann_m:.2f} ans")
    k8.metric("Sinistre max observé",   fmt_eur(df["CHARGE"].max()))

    st.divider()

    c1, c2 = st.columns([1, 1.3])

    with c1:
        act_cnt = (
            df.groupby("ACT_LABEL").size()
            .reset_index(name="Contrats")
            .sort_values("Contrats", ascending=False)
        )
        fig = px.pie(
            act_cnt, names="ACT_LABEL", values="Contrats",
            title="Répartition du portefeuille par activité",
            color_discrete_sequence=PALETTE, hole=0.45,
        )
        fig.update_traces(textinfo="percent",textposition="inside")

        fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5
    ),
    showlegend=True) 
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sp_act = get_sp_data(df, df_t, "ACTIVIT2")
        sp_act["Activité"] = sp_act["ACTIVIT2"].map(ACT_LABELS).fillna(sp_act["ACTIVIT2"])
        sp_v = sp_act[
            sp_act["sinistres_moy"] + sp_act["prime_pred_moy"] > 0
        ].sort_values("sp_pct", ascending=False)

        fig = go.Figure(go.Bar(
            x=sp_v["Activité"], y=sp_v["sp_pct"],
            marker_color=sp_v["sp_pct"].apply(sp_color).tolist(),
            text=sp_v["sp_pct"].map("{:.0f}%".format),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>S/P = %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=100, line_dash="dash", line_color=RED, line_width=1.5,
                      annotation_text="Seuil 100%", annotation_font_color=RED,
                      annotation_position="top right")
        fig.add_hline(y=70, line_dash="dot", line_color=AMBER, line_width=1,
                      annotation_text="Cible 70%", annotation_font_color=AMBER)
        fig.update_layout(**THEME, title="Ratio S/P (%) par activité",
                          yaxis_title="S/P (%)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    top_z = (
        df.groupby("ZONE")["CHARGE"].mean()
        .reset_index()
        .rename(columns={"CHARGE": "Charge moy. (€)"})
        .nlargest(12, "Charge moy. (€)")
    )
    fig = px.bar(
        top_z, x="ZONE", y="Charge moy. (€)",
        title="Top 12 zones — charge sinistre moyenne",
        color="Charge moy. (€)", color_continuous_scale="OrRd",
        text="Charge moy. (€)", labels={"ZONE": "Zone"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}€", textposition="outside")
    fig.update_layout(**THEME, coloraxis_showscale=False, xaxis_type="category")
    st.plotly_chart(fig, use_container_width=True)

    sp_label = "rentable" if sp_glob < 100 else "déficitaire"
    st.info(
        f"**Synthèse :** {n_c:,} contrats — {n_sin} sinistres ({tx_sin:.2f}%). "
        f"S/P global = **{sp_glob:.1f}%** (portefeuille **{sp_label}**). "
        f"Prime pure moy. prédite : **{prime_p:.1f}€**. "
        "Zones 5, 70, 49 surexposées."
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 – PORTEFEUILLE
# ═══════════════════════════════════════════════════════════════
elif page == "Portefeuille":
    page_title(
        "Structure du Portefeuille",
        "Composition, segmentation et profil tarifaire des contrats Multirisque Agricole",
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Contrats sélectionnés", f"{len(df):,}")
    k2.metric("Exposition moy.",        f"{df['ANNEE_ASSURANCE'].mean():.2f} ans")
    k3.metric("Ancienneté moy.",        f"{df['ANCIENNETE'].mean():.1f} ans")
    k4.metric("Charge moy.",            f"{df['CHARGE'].mean():.1f} €")

    st.divider()

    pc1, pc2 = st.columns(2)

    with pc1:
        tail_cnt = df["TAILLE1"].fillna("N/A").value_counts().reset_index()
        tail_cnt.columns = ["Tranche capital", "n"]
        tail_cnt = tail_cnt.sort_values("Tranche capital")
        fig = px.pie(
            tail_cnt, names="Tranche capital", values="n",
            title="Répartition par taille de contrat (TAILLE1)",
            color_discrete_sequence=PALETTE, hole=0.40,
        )
        fig.update_traces(textinfo="label+percent", textfont_size=10)
        fig.update_layout(**THEME, showlegend=False, height=330)
        st.plotly_chart(fig, use_container_width=True)

    with pc2:
        anc_cnt = df["ANCIE_BIN"].value_counts().reset_index()
        anc_cnt.columns = ["Ancienneté", "n"]
        fig = px.pie(
            anc_cnt, names="Ancienneté", values="n",
            title="Répartition par ancienneté du contrat",
            color_discrete_sequence=PALETTE, hole=0.40,
        )
        fig.update_traces(textinfo="label+percent", textfont_size=10)
        fig.update_layout(**THEME, showlegend=False, height=330)
        st.plotly_chart(fig, use_container_width=True)

    bc1, bc2 = st.columns(2)

    with bc1:
        anc_ord = ["0 an", "1-2 ans", "3-5 ans", "6-10 ans", "10+ ans"]
        anc_df = (
            df.groupby("ANCIE_BIN")
            .agg(taux=("SINISTRE", "mean"), charge_moy=("CHARGE", "mean"))
            .reset_index()
        )
        anc_df["ANCIE_BIN"] = pd.Categorical(
            anc_df["ANCIE_BIN"], categories=anc_ord, ordered=True
        )
        anc_df = anc_df.sort_values("ANCIE_BIN")

        fig = go.Figure()
        fig.add_bar(
            x=anc_df["ANCIE_BIN"], y=anc_df["taux"] * 100,
            name="Taux sin. (%)", marker_color=ORANGE, opacity=0.8,
            text=anc_df["taux"].map("{:.2%}".format), textposition="outside",
        )
        fig.add_scatter(
            x=anc_df["ANCIE_BIN"], y=anc_df["charge_moy"],
            name="Charge moy. (€)", yaxis="y2",
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5), marker=dict(size=9),
        )
        fig.update_layout(
            **THEME, title="Sinistralité & Charge selon l'ancienneté",
            yaxis=dict(title="Taux sin. (%)"),
            yaxis2=dict(title="Charge moy. (€)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.25), height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with bc2:
        tail_c = (
            df.groupby("TAILLE1")
            .agg(charge_moy=("CHARGE", "mean"), taux=("SINISTRE", "mean"), n=("ID", "count"))
            .reset_index()
            .sort_values("TAILLE1")
        )
        fig = go.Figure()
        fig.add_bar(
            x=tail_c["TAILLE1"], y=tail_c["charge_moy"],
            name="Charge moy. (€)", marker_color=BLUE, opacity=0.8,
            text=tail_c["charge_moy"].map("{:.0f}€".format), textposition="outside",
        )
        fig.add_scatter(
            x=tail_c["TAILLE1"], y=tail_c["taux"] * 100,
            name="Taux sin. (%)", yaxis="y2",
            mode="lines+markers",
            line=dict(color=ORANGE, width=2.5), marker=dict(size=9),
        )
        fig.update_layout(
            **THEME, title="Charge & Sinistralité par taille de contrat",
            xaxis_tickangle=-30,
            yaxis=dict(title="Charge moy. (€)"),
            yaxis2=dict(title="Taux sin. (%)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.3), height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Tableau récapitulatif par activité")
    tbl = df.groupby("ACTIVIT2").agg(
        Contrats=("ID", "count"),
        Charge_tot=("CHARGE", "sum"),
        Charge_moy=("CHARGE", "mean"),
        Freq_moy=("FREQ", "mean"),
        Taux_sin=("SINISTRE", "mean"),
        Expo_moy=("ANNEE_ASSURANCE", "mean"),
    ).reset_index()
    tbl["Activité"]     = tbl["ACTIVIT2"].map(ACT_LABELS)
    tbl["Part portef."] = (tbl["Contrats"] / tbl["Contrats"].sum() * 100).map("{:.1f}%".format)
    tbl["Taux sin."]    = (tbl["Taux_sin"] * 100).map("{:.2f}%".format)
    tbl["Charge moy."]  = tbl["Charge_moy"].map("{:.1f}€".format)
    tbl["Charge tot."]  = tbl["Charge_tot"].map("{:,.0f}€".format)
    tbl["Fréq. moy."]   = tbl["Freq_moy"].map("{:.5f}".format)
    tbl["Expo moy."]    = tbl["Expo_moy"].map("{:.2f}".format)

    q25 = tbl["Charge_moy"].quantile(0.25)
    q75 = tbl["Charge_moy"].quantile(0.75)

    def risk_label(v):
        if v >= q75:
            return "Élevé"
        if v >= q25:
            return "Modéré"
        return "Faible"

    tbl["Risque"] = tbl["Charge_moy"].apply(risk_label)
    st.dataframe(
        tbl[["ACTIVIT2", "Activité", "Contrats", "Part portef.", "Expo moy.",
             "Fréq. moy.", "Charge moy.", "Charge tot.", "Taux sin.", "Risque"]],
        use_container_width=True, hide_index=True,
    )

    st.info(
        "**Lecture :** Pics de sinistralité à 2 ans et 10+ ans d'ancienneté. "
        "Capital élevé (>1M€) = sévérité plus forte. "
        "ACT4 (horticulture) : taux modéré mais coût extrême."
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 – SINISTRALITÉ
# ═══════════════════════════════════════════════════════════════
elif page == "Sinistralité":
    page_title(
        "Analyse de la Sinistralité",
        "Distribution, sévérité et drivers du risque incendie MRA",
    )

    n_sin = int(df["SINISTRE"].sum())
    tx    = df["SINISTRE"].mean() * 100
    st.warning(
        f"**Contexte :** {n_sin} sinistres / {len(df):,} contrats ({tx:.2f}%) "
        "— Distribution heavy-tail. Justifie le modèle Poisson × Tweedie."
    )

    d1, d2 = st.columns(2)

    with d1:
        sin_df = df[df["CHARGE"] > 0]
        fig = px.histogram(
            sin_df, x="LOG_CHARGE", nbins=45,
            title=f"Distribution log(CHARGE+1) — {len(sin_df)} sinistres",
            color_discrete_sequence=[RED], opacity=0.80,
            labels={"LOG_CHARGE": "log(Charge + 1)"},
        )
        fig.update_layout(**THEME, yaxis_title="Fréquence", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        percs = [50, 75, 90, 95, 99, 99.5]
        vals  = [np.percentile(df["CHARGE"], p) for p in percs]
        pct   = pd.DataFrame({"Percentile": [f"P{p}" for p in percs], "Charge (€)": vals})
        fig = go.Figure(go.Bar(
            x=pct["Percentile"], y=pct["Charge (€)"],
            marker_color=[GREEN, GREEN, AMBER, AMBER, RED, RED],
            text=pct["Charge (€)"].map("{:,.0f}€".format),
            textposition="outside",
        ))
        fig.update_layout(**THEME, title="Percentiles de la charge sinistre",
                          yaxis_title="Charge (€)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    s1, s2 = st.columns(2)

    with s1:
        nb_g = (
            df.assign(NBS=df["NBSINSTRT"].fillna(0).clip(0, 4).astype(int))
            .groupby("NBS")
            .agg(charge_moy=("CHARGE", "mean"), taux=("SINISTRE", "mean"))
            .reset_index()
        )
        fig = go.Figure()
        fig.add_bar(
            x=nb_g["NBS"], y=nb_g["charge_moy"],
            name="Charge moy. (€)", marker_color=ORANGE,
            text=nb_g["charge_moy"].map("{:.0f}€".format), textposition="outside",
        )
        fig.add_scatter(
            x=nb_g["NBS"], y=nb_g["taux"] * 100,
            name="Taux sin. (%)", yaxis="y2",
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5), marker=dict(size=9),
        )
        fig.update_layout(
            **THEME, title="Impact des sinistres passés (NBSINSTRT)",
            xaxis_title="Nb sinistres passés",
            yaxis=dict(title="Charge moy. (€)"),
            yaxis2=dict(title="Taux sin. (%)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.25), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with s2:
        car1 = (
            df.groupby("CARACT1")["CHARGE"]
            .agg(charge_moy="mean", taux=lambda x: (x > 0).mean())
            .reset_index()
        )
        car1["Lib"] = car1["CARACT1"].map(
            {"N": "Standard (N)", "R": "Structure risque (R)", "O": "Autre (O)"}
        ).fillna(car1["CARACT1"])

        fig = go.Figure()
        fig.add_bar(
            x=car1["Lib"], y=car1["charge_moy"],
            name="Charge moy. (€)",
            marker_color=[RED if c == "R" else BLUE for c in car1["CARACT1"]],
            text=car1["charge_moy"].map("{:.1f}€".format), textposition="outside",
        )
        fig.add_scatter(
            x=car1["Lib"], y=car1["taux"] * 100,
            name="Taux sin. (%)", yaxis="y2",
            mode="markers", marker=dict(color=AMBER, size=18, symbol="diamond"),
        )
        fig.update_layout(
            **THEME, title="Type de bâtiment (CARACT1) — Charge & Taux",
            yaxis=dict(title="Charge moy. (€)"),
            yaxis2=dict(title="Taux sin. (%)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.25), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    z_sin = (
        df.groupby("ZONE")
        .agg(charge_moy=("CHARGE", "mean"), taux=("SINISTRE", "mean"), n=("ID", "count"))
        .reset_index()
        .sort_values("charge_moy", ascending=False)
        .head(15)
    )
    fig = px.bar(
        z_sin, x="ZONE", y="charge_moy",
        color="charge_moy", color_continuous_scale="YlOrRd",
        title="Top 15 zones les plus sinistrées (charge moy. observée)",
        text="charge_moy", labels={"ZONE": "Zone", "charge_moy": "Charge moy. (€)"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}€", textposition="outside")
    fig.update_layout(**THEME, coloraxis_showscale=False, xaxis_type="category")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Insights :** Sinistres passés = meilleur prédicteur individuel (NBSINSTRT). "
        "Bâtiments à structure risque (CARACT1=R) : charge 2× supérieure. "
        "Zones 5, 70, 49 = concentration du risque extrême."
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 – RÉEL VS PRÉDIT
# ═══════════════════════════════════════════════════════════════
elif page == "Écarts modèle & performance":
    page_title(
        "Réel vs Prédit",
        "Comparaison sinistres observés (train) vs primes pures prédites (test) par le modèle",
    )

    st.info(
        "**Méthodologie :** Sinistres réels (train) = CHARGE. "
        "Primes pures prédites (test) = CHARGE_PRED du modèle LightGBM. "
        "S/P = Sinistres / Primes — mesure la couverture par segment."
    )

    sp_act = get_sp_data(df, df_t, "ACTIVIT2")
    sp_act["Activité"] = sp_act["ACTIVIT2"].map(ACT_LABELS).fillna(sp_act["ACTIVIT2"])
    sp_valid = sp_act[sp_act["sinistres_moy"] + sp_act["prime_pred_moy"] > 0]

    v1, v2 = st.columns(2)

    with v1:
        fig = go.Figure()
        fig.add_bar(
            name="Sinistres réels (€)", x=sp_valid["Activité"],
            y=sp_valid["sinistres_moy"], marker_color=RED, opacity=0.8,
            text=sp_valid["sinistres_moy"].map("{:.1f}€".format), textposition="outside",
        )
        fig.add_bar(
            name="Prime pure prédite (€)", x=sp_valid["Activité"],
            y=sp_valid["prime_pred_moy"], marker_color=BLUE, opacity=0.8,
            text=sp_valid["prime_pred_moy"].map("{:.1f}€".format), textposition="outside",
        )
        fig.update_layout(
            **THEME, title="Charge réelle vs Prime prédite — par activité",
            barmode="group", yaxis_title="€ moyen / contrat",
            legend=dict(orientation="h", y=-0.3), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with v2:
        fig = go.Figure()
        fig.add_bar(
            name="Fréquence réelle", x=sp_valid["Activité"],
            y=sp_valid["freq_reel"], marker_color=ORANGE, opacity=0.8,
            text=sp_valid["freq_reel"].map("{:.5f}".format), textposition="outside",
        )
        fig.add_bar(
            name="Fréquence prédite", x=sp_valid["Activité"],
            y=sp_valid["freq_pred"], marker_color=TEAL, opacity=0.8,
            text=sp_valid["freq_pred"].map("{:.5f}".format), textposition="outside",
        )
        fig.update_layout(
            **THEME, title="Fréquence réelle vs Prédite — par activité",
            barmode="group", yaxis_title="Fréquence",
            legend=dict(orientation="h", y=-0.3), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    sp_zone = get_sp_data(df, df_t, "ZONE")
    sp_zone = (
        sp_zone[sp_zone["n_train"] >= 20]
        .sort_values("sp_pct", ascending=False)
        .head(20)
    )

    fig = go.Figure()
    fig.add_bar(
        name="Charge réelle (€)", x=sp_zone["ZONE"].astype(str),
        y=sp_zone["sinistres_moy"], marker_color=RED, opacity=0.8,
    )
    fig.add_bar(
        name="Prime prédite (€)", x=sp_zone["ZONE"].astype(str),
        y=sp_zone["prime_pred_moy"], marker_color=BLUE, opacity=0.8,
    )
    fig.add_scatter(
        x=sp_zone["ZONE"].astype(str), y=sp_zone["sp_pct"],
        name="S/P (%)", yaxis="y2", mode="lines+markers",
        line=dict(color=AMBER, width=2), marker=dict(size=7),
    )
    fig.add_hline(y=100, line_dash="dash", line_color=RED, line_width=1.5, yref="y2",
                  annotation_text="Seuil 100%", annotation_font_color=RED)
    fig.update_layout(
        **THEME,
        title="Top 20 des zones géographiques les plus déficitaires (Rapport S/P le plus élevé)",
        barmode="group", xaxis_title="Zone",
        yaxis=dict(title="€ moyen"),
        yaxis2=dict(title="S/P (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.25), height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Tableau des écarts Réel vs Prédit par activité")
    rows = []
    for _, row in sp_valid.iterrows():
        cl, lb = sp_badge(row["sp_pct"])
        rows.append({
            "Code":               row["ACTIVIT2"],
            "Activité":           row["Activité"],
            "N train":            f"{int(row['n_train']):,}",
            "N test":             f"{int(row['n_test']):,}",
            "Sinistr. réel (€)":  f"{row['sinistres_moy']:.1f}",
            "Prime prédite (€)":  f"{row['prime_pred_moy']:.1f}",
            "Écart (€)":          f"{row['ecart_moy']:+.1f}",
            "Fréq. réelle":       f"{row['freq_reel']:.5f}",
            "Fréq. prédite":      f"{row['freq_pred']:.5f}",
            "S/P (%)":            f"{row['sp_pct']:.1f}%",
            "Statut":             lb,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "**Interprétation :** S/P < 70% = portefeuille très rentable. "
        "S/P > 100% = sinistres > primes → segment déficitaire à reprimer. "
        "Écart de fréquence positif = sous-estimation par le modèle."
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 – RENTABILITÉ S/P
# ═══════════════════════════════════════════════════════════════
elif page == "Rentabilité technique":
    page_title(
        "Rentabilité par Segment — Ratio S/P",
         "Analyse du ratio sinistres / primes (S/P) — pilotage de la performance et de la tarification",
    )

    tot_sin = df["CHARGE"].sum()
    tot_pri = df_t["CHARGE_PRED"].sum() if len(df_t) > 0 else 1
    sp_glob = tot_sin / (tot_pri + 1e-9) * 100
    prime_m = df_t["CHARGE_PRED"].mean() if len(df_t) > 0 else 0
    moy_sin = df["CHARGE"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("S/P Global",              f"{sp_glob:.1f}%",
              delta="Rentable" if sp_glob < 100 else "Déficitaire",
              delta_color="normal" if sp_glob < 100 else "inverse")
    k2.metric("Total sinistres",          fmt_eur(tot_sin))
    k3.metric("Prime pure moy. prédite", f"{prime_m:.1f} €")
    k4.metric("Écart moy./contrat",       f"{moy_sin - prime_m:+.1f} €",
              delta_color="inverse" if moy_sin > prime_m else "normal")

    st.divider()

    sp_act = get_sp_data(df, df_t, "ACTIVIT2")
    sp_act["Activité"] = sp_act["ACTIVIT2"].map(ACT_LABELS).fillna(sp_act["ACTIVIT2"])
    sp_act = sp_act[sp_act["n_train"] > 0].sort_values("sp_pct", ascending=False)

    fig = go.Figure()
    for _, row in sp_act.iterrows():
        fig.add_bar(
            x=[row["Activité"]], y=[row["sp_pct"]],
            marker_color=sp_color(row["sp_pct"]),
            text=f"{row['sp_pct']:.0f}%", textposition="outside",
            hovertemplate=(
                f"<b>{row['Activité']}</b><br>"
                f"S/P = {row['sp_pct']:.1f}%<br>"
                f"Sinistr. = {row['sinistres_moy']:.1f}€<br>"
                f"Prime = {row['prime_pred_moy']:.1f}€<extra></extra>"
            ),
            name=row["Activité"],
        )
    fig.add_hline(y=100, line_dash="dash", line_color=RED, line_width=2,
                  annotation_text="Seuil 100% (équilibre)",
                  annotation_font_color=RED, annotation_position="top left")
    fig.add_hline(y=70, line_dash="dot", line_color=AMBER, line_width=1.5,
                  annotation_text="Cible 70% (rentabilité)",
                  annotation_font_color=AMBER, annotation_position="top left")
    fig.add_hrect(y0=0,   y1=70,  fillcolor="rgba(46,125,50,.06)",  line_width=0)
    fig.add_hrect(y0=70,  y1=100, fillcolor="rgba(245,127,23,.06)", line_width=0)
    max_sp = max(sp_act["sp_pct"].max() * 1.1, 110)
    fig.add_hrect(y0=100, y1=max_sp, fillcolor="rgba(198,40,40,.06)", line_width=0)
    fig.update_layout(
        **THEME, title="Ratio S/P (%) par activité agricole",
        yaxis_title="S/P (%)", showlegend=False, height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    sc1, sc2 = st.columns(2)

    with sc1:
        fig = px.scatter(
            sp_act,
            x="prime_pred_moy", y="sinistres_moy", size="n_train",
            color="Activité",
            hover_name="Activité",
            hover_data={"sp_pct": ":.1f", "n_train": True},
            title="Prime prédite vs Sinistres réels (taille = volume)",
            labels={
                "prime_pred_moy": "Prime pure prédite (€)",
                "sinistres_moy":  "Sinistres réels (€)",
            },
            size_max=55,
            color_discrete_sequence=PALETTE,
        )
        mx = max(sp_act[["prime_pred_moy", "sinistres_moy"]].max().max() * 1.1, 50)
        fig.add_scatter(
            x=[0, mx], y=[0, mx], mode="lines",
            name="S/P = 100%",
            line=dict(color=RED, dash="dash", width=1.5),
        )
        fig.update_layout(**THEME, legend=dict(orientation="h", y=-0.3), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        sp_zone = get_sp_data(df, df_t, "ZONE")
        sp_zone = (
            sp_zone[sp_zone["n_train"] >= 15]
            .sort_values("sp_pct", ascending=False)
            .head(15)
        )
        fig = go.Figure(go.Bar(
            x=sp_zone["ZONE"].astype(str), y=sp_zone["sp_pct"],
            marker_color=sp_zone["sp_pct"].apply(sp_color).tolist(),
            text=sp_zone["sp_pct"].map("{:.0f}%".format),
            textposition="outside",
            hovertemplate="Zone %{x}<br>S/P = %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=100, line_dash="dash", line_color=RED, line_width=1.5)
        fig.add_hline(y=70,  line_dash="dot",  line_color=AMBER, line_width=1)
        fig.update_layout(
            **THEME, title="Ratio S/P par zone géographique (top 15)",
            xaxis_title="Zone", yaxis_title="S/P (%)", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Matrice de rentabilité — Statut par activité")
    for _, row in sp_act[sp_act["n_train"] > 0].iterrows():
        cl, lb = sp_badge(row["sp_pct"])
        cols = st.columns([2, 1.4, 1.4, 1.4, 1.4, 1.8])
        cols[0].write(f"**{row['Activité']}**")
        cols[1].metric("Sinistres", f"{row['sinistres_moy']:.1f}€")
        cols[2].metric("Prime",     f"{row['prime_pred_moy']:.1f}€")
        cols[3].metric("S/P",       f"{row['sp_pct']:.1f}%")
        cols[4].metric("Taux sin.", f"{row['taux_sin'] * 100:.2f}%")
        cols[5].markdown(f"<span class='{cl}'>{lb}</span>", unsafe_allow_html=True)

    st.divider()

    sp_anc = get_sp_data(df, df_t, "ANCIE_BIN")
    anc_ord = ["0 an", "1-2 ans", "3-5 ans", "6-10 ans", "10+ ans"]
    sp_anc["ANCIE_BIN"] = pd.Categorical(
        sp_anc["ANCIE_BIN"], categories=anc_ord, ordered=True
    )
    sp_anc = sp_anc.sort_values("ANCIE_BIN").dropna(subset=["ANCIE_BIN"])

    fig = go.Figure()
    fig.add_bar(x=sp_anc["ANCIE_BIN"].astype(str), y=sp_anc["sinistres_moy"],
                name="Sinistres réels", marker_color=RED, opacity=0.8)
    fig.add_bar(x=sp_anc["ANCIE_BIN"].astype(str), y=sp_anc["prime_pred_moy"],
                name="Prime prédite", marker_color=BLUE, opacity=0.8)
    fig.add_scatter(
        x=sp_anc["ANCIE_BIN"].astype(str), y=sp_anc["sp_pct"],
        name="S/P (%)", yaxis="y2", mode="lines+markers",
        line=dict(color=AMBER, width=2.5), marker=dict(size=10, color=AMBER),
    )
    fig.add_hline(y=100, line_dash="dash", line_color=RED, line_width=1.5, yref="y2")
    fig.update_layout(
        **THEME, title="Rentabilité S/P selon l'ancienneté du contrat",
        barmode="group",
        yaxis=dict(title="€ moy."),
        yaxis2=dict(title="S/P (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.25), height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Pilotage tarifaire :** S/P < 70% → portefeuille très rentable. "
        "S/P 70-100% → zone de rentabilité cible. "
        "S/P > 100% → révision de prime à la hausse. "
        "ACT4 et zones 5/70/49 nécessitent un ajustement tarifaire immédiat."
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 6 – PRÉDICTIONS MODÈLE
# ═══════════════════════════════════════════════════════════════
elif page == "Modélisation & projections":
    page_title(
    "Modélisation & Projections",
    "Modèle LightGBM : fréquence (Poisson) × coût (Tweedie) — validation croisée 5 folds | 373 variables explicatives",
)

    n_te  = len(df_t)
    moy_p = df_t["CHARGE_PRED"].mean()
    fq_p  = df_t["FREQ_PRED"].mean()
    cm_p  = (
        df_t.loc[df_t["CM_PRED"] > 0, "CM_PRED"].mean()
        if (df_t["CM_PRED"] > 0).any() else 0
    )
    sp_g  = (df["CHARGE"].mean() / (moy_p + 1e-9) * 100) if moy_p > 0 else 0

    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    pk1.metric("Contrats test",           f"{n_te:,}")
    pk2.metric("Prime pure moy. prédite", f"{moy_p:.1f} €")
    pk3.metric("Fréq. moy. prédite",      f"{fq_p:.5f}")
    pk4.metric("CM moyen prédit",         fmt_eur(cm_p))
    pk5.metric("RMSE OOF",               "6 802 €",
               delta="σ = ±328€ / fold", delta_color="off")

    st.divider()

    r1a, r1b = st.columns(2)

    with r1a:
        fig = go.Figure(go.Histogram(
            x=np.log1p(df_t["CHARGE_PRED"]), nbinsx=50,
            marker_color=BLUE, opacity=0.75,
        ))
        fig.update_layout(
            **THEME, title="Distribution log(Prime prédite + 1) — jeu de test",
            xaxis_title="log(CHARGE_PRED + 1)",
            yaxis_title="Fréquence", height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        zone_p = (
            df_t.groupby("ZONE")["CHARGE_PRED"]
            .sum()
            .reset_index()
            .rename(columns={"CHARGE_PRED": "Prime totale prédite (€)"})
            .nlargest(12, "Prime totale prédite (€)")
        )
        fig = px.bar(
            zone_p, x="ZONE", y="Prime totale prédite (€)",
            title="Prime totale prédite — Top 12 zones",
            color="Prime totale prédite (€)", color_continuous_scale="Blues_r",
            text="Prime totale prédite (€)", labels={"ZONE": "Zone"},
        )
        fig.update_traces(texttemplate="%{text:,.0f}€", textposition="outside")
        fig.update_layout(**THEME, coloraxis_showscale=False,
                          xaxis_type="category", height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Performances cross-validation — RMSE par fold")

    fold_df  = pd.DataFrame({"Fold": list(RMSE_FOLDS.keys()),
                              "RMSE (€)": list(RMSE_FOLDS.values())})
    rmse_std = np.std(list(RMSE_FOLDS.values()))

    fig = go.Figure()
    fig.add_bar(
        x=fold_df["Fold"], y=fold_df["RMSE (€)"],
        marker_color=[
            RED if v == max(fold_df["RMSE (€)"]) else BLUE
            for v in fold_df["RMSE (€)"]
        ],
        text=fold_df["RMSE (€)"].map("{:,.0f}€".format),
        textposition="outside",
    )
    fig.add_hline(
        y=6802.46, line_dash="dash", line_color=ORANGE, line_width=2,
        annotation_text=f"  RMSE OOF global = 6 802€ (σ = {rmse_std:.0f}€)",
        annotation_font_color=ORANGE,
    )
    fig.update_layout(
        **THEME, title="Stabilité du modèle LightGBM — RMSE par fold",
        yaxis=dict(title="RMSE (€)", range=[5_800, 7_600]), height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Architecture")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("""
**Modèle FREQ — Poisson**
```
Distribution : Poisson
Lien         : Log
Num. leaves  : 127
Learning rate: 0.05
Feature frac.: 80 %
Early stop   : 100
```""")
    with ac2:
        st.markdown("""
**Modèle CM — Tweedie**
```
Distribution : Tweedie (p=1.5)
Lien         : Log
Entraîné sur : sinistres > 0
Num. leaves  : 127
Learning rate: 0.05
Feature frac.: 80 %
```""")
    with ac3:
        st.markdown("""
**Validation croisée**
```
Algorithme : LightGBM
Folds      : 5-Fold CV
Seed       : 42
Features   : 373
Métrique   : RMSE (CHARGE)
OOF RMSE   : 6 802 €
```""")

    st.success(
        f"**Modèle stable** — variance inter-fold σ = {rmse_std:.0f}€ "
        f"({rmse_std / 6802 * 100:.1f}% de la RMSE globale). "
        f"Prime pure moy. prédite : **{moy_p:.1f}€** — "
        f"Ratio S/P indicatif : **{sp_g:.1f}%**."
    )


# python -m streamlit run dashboard.py 
#  cd "C:\Users\User\Desktop\Projets\CA Assurance"