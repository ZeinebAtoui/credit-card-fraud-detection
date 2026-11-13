

import os

from pathlib import Path

import warnings

import joblib

import numpy as np

import pandas as pd

import streamlit as st

import plotly.express as px

import plotly.graph_objects as go



from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    average_precision_score,

    confusion_matrix,

    roc_curve,

    precision_recall_curve

)



warnings.filterwarnings("ignore")







st.set_page_config(

    page_title="Fraud Detection Dashboard",

   

    layout="wide",

    initial_sidebar_state="expanded"

)






BASE_DIR = Path(__file__).resolve().parent


MODEL_DIR = BASE_DIR / "models"


DATA_DIR = BASE_DIR


RESULTS_DIR = BASE_DIR



MODEL_FILES = {

    "Isolation Forest": "best_isolation_forest.joblib",

    "HBOS": "best_hbos.joblib",

    "SGD One-Class SVM": "best_sgd_one_class_svm.joblib",

    "Random Forest A": "random_forest_A.joblib",

    "Random Forest B": "random_forest_B.joblib",

    "XGBoost A": "xgboost_A.joblib",

    "XGBoost B": "xgboost_B.joblib",

    "Logistic Regression A": "logistic_regression_A.joblib",

    "Logistic Regression B": "logistic_regression_B.joblib",

    "Modèle hybride": "hybrid_fraud_model.joblib"

}



UNSUPERVISED_MODELS = [

    "Isolation Forest",

    "HBOS",

    "SGD One-Class SVM"

]



SUPERVISED_MODELS = [

    "Random Forest A",

    "Random Forest B",

    "XGBoost A",

    "XGBoost B",

    "Logistic Regression A",

    "Logistic Regression B"

]







st.markdown("""

<style>

    .main {

        background-color: #F7F9FC;

    }



    h1, h2, h3 {

        color: #17365D;

    }



    .stMetric {

        background-color: white;

        border: 1px solid #E3EAF2;

        padding: 15px;

        border-radius: 12px;

    }



    div[data-testid="stSidebar"] {

        background-color: #EDF2F8;

    }



    .subtitle {

        color: #64748B;

        font-size: 16px;

    }



    .section-title {

        color: #17365D;

        font-weight: 700;

        font-size: 22px;

        margin-top: 15px;

    }



    .info-card {

        background-color: white;

        padding: 18px;

        border-radius: 12px;

        border: 1px solid #E3EAF2;

    }

</style>

""", unsafe_allow_html=True)









@st.cache_data

def load_data():

    data = {}



    files = {

        "X_val": "X_val.npy",

        "y_val": "y_val.npy",

        "X_test": "X_test.npy",

        "y_test": "y_test.npy"

    }



    for key, filename in files.items():

        path = DATA_DIR / filename



        if not os.path.exists(path):

            raise FileNotFoundError(

                f"Fichier introuvable : {path}"

            )



        data[key] = np.load(path)



    return data








@st.cache_resource

def load_models():

    models = {}



    for name, filename in MODEL_FILES.items():

        path = MODEL_DIR / filename



        if os.path.exists(path):

            try:

                models[name] = joblib.load(path)

            except Exception as error:

                st.warning(

                    f"Impossible de charger {name} : {error}"

                )



    return models









def extract_model(artifact):

    
   



    if not isinstance(artifact, dict):

        return artifact



    possible_keys = [

        "model",

        "estimator",

        "clf",

        "detector",

        "best_model",

        "pipeline"

    ]



    for key in possible_keys:

        if key in artifact:

            return artifact[key]



    return artifact





def get_threshold(artifact, default=0.5):




    if isinstance(artifact, dict):

        possible_keys = [

            "threshold",

            "best_threshold",

            "optimal_threshold",

            "selected_threshold"

        ]



        for key in possible_keys:

            if key in artifact:

                try:

                    return float(artifact[key])

                except (TypeError, ValueError):

                    pass



    return default





def get_unsupervised_scores(model, model_name, X):

   



    if hasattr(model, "decision_function"):

        scores = model.decision_function(X)



    elif hasattr(model, "score_samples"):

        scores = model.score_samples(X)



    else:

        raise ValueError(

            f"{model_name} ne possède pas de méthode de scoring."

        )



    scores = np.asarray(scores).reshape(-1)



    if model_name in ["Isolation Forest", "SGD One-Class SVM"]:

        scores = -scores



    return scores





def get_supervised_scores(model, X):

    


    if hasattr(model, "predict_proba"):

        scores = model.predict_proba(X)



        if scores.ndim == 2:

            if scores.shape[1] > 1:

                scores = scores[:, 1]

            else:

                scores = scores[:, 0]



    elif hasattr(model, "decision_function"):

        scores = model.decision_function(X)



    else:

        scores = model.predict(X)



    return np.asarray(scores).reshape(-1)





def percentile_normalization(scores, reference_scores):

    



    reference_scores = np.asarray(reference_scores).ravel()

    scores = np.asarray(scores).ravel()



    reference_scores = np.sort(reference_scores)



    if len(reference_scores) == 0:

        raise ValueError("La distribution de référence est vide.")



    return (

        np.searchsorted(

            reference_scores,

            scores,

            side="right"

        ) / len(reference_scores)

    )





def get_hybrid_scores(artifact, X):

    



    if not isinstance(artifact, dict):

        raise ValueError(

            "Le fichier du modèle hybride doit contenir un dictionnaire."

        )



    unsup_model = artifact["unsupervised_model"]

    sup_model = artifact["supervised_model"]



    unsup_name = artifact["unsupervised_name"]

    alpha = float(artifact["alpha"])



    unsup_reference = artifact["unsupervised_reference_scores"]

    sup_reference = artifact["supervised_reference_scores"]



    unsup_scores = get_unsupervised_scores(

        unsup_model,

        unsup_name,

        X

    )



    sup_scores = get_supervised_scores(

        sup_model,

        X

    )



    unsup_norm = percentile_normalization(

        unsup_scores,

        unsup_reference

    )



    sup_norm = percentile_normalization(

        sup_scores,

        sup_reference

    )



    hybrid_scores = (

        alpha * unsup_norm

        + (1 - alpha) * sup_norm

    )



    return hybrid_scores









def sample_dataset(X, y, max_samples):




    n = len(y)



    if n <= max_samples:

        return X, y



    rng = np.random.default_rng(42)



    fraud_indices = np.where(y == 1)[0]

    normal_indices = np.where(y == 0)[0]



    fraud_ratio = len(fraud_indices) / n



    n_fraud = min(

        len(fraud_indices),

        max(1, int(max_samples * fraud_ratio))

    )



    n_normal = min(

        len(normal_indices),

        max_samples - n_fraud

    )



   

    remaining = max_samples - n_fraud - n_normal



    if remaining > 0:

        extra_fraud = min(

            remaining,

            len(fraud_indices) - n_fraud

        )

        n_fraud += extra_fraud

        remaining -= extra_fraud



    selected_fraud = rng.choice(

        fraud_indices,

        size=n_fraud,

        replace=False

    )



    selected_normal = rng.choice(

        normal_indices,

        size=n_normal,

        replace=False

    )



    indices = np.concatenate([

        selected_fraud,

        selected_normal

    ])



    rng.shuffle(indices)



    return X[indices], y[indices]








def calculate_model_scores(models, X):

    scores = {}

    errors = {}



    for name, artifact in models.items():



        try:

            if name in UNSUPERVISED_MODELS:

                model = extract_model(artifact)



                scores[name] = get_unsupervised_scores(

                    model,

                    name,

                    X

                )



            elif name in SUPERVISED_MODELS:

                model = extract_model(artifact)



                scores[name] = get_supervised_scores(

                    model,

                    X

                )



            elif name == "Modèle hybride":

                scores[name] = get_hybrid_scores(

                    artifact,

                    X

                )



        except Exception as error:

            errors[name] = str(error)



    return scores, errors









def calculate_metrics(y_true, scores, threshold):

    y_pred = (scores >= threshold).astype(int)



    tn, fp, fn, tp = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]

    ).ravel()



    accuracy = accuracy_score(y_true, y_pred)



    precision = precision_score(

        y_true,

        y_pred,

        zero_division=0

    )



    recall = recall_score(

        y_true,

        y_pred,

        zero_division=0

    )



    f1 = f1_score(

        y_true,

        y_pred,

        zero_division=0

    )



    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(y_true, scores)

        pr_auc = average_precision_score(y_true, scores)

    else:

        roc_auc = np.nan

        pr_auc = np.nan



    alert_rate = np.mean(y_pred)



    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-score": f1,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc,

        "Alert Rate": alert_rate,

        "TP": int(tp),

        "FP": int(fp),

        "TN": int(tn),

        "FN": int(fn),

        "Threshold": threshold

    }









st.title("💳 Fraud Detection Dashboard")



st.markdown(

    '<p class="subtitle">'

    'Analyse, comparaison et évaluation des modèles '

    'de détection de fraude bancaire.'

    '</p>',

    unsafe_allow_html=True

)



try:

    data = load_data()

    models = load_models()



except Exception as error:

    st.error(f"Erreur de chargement : {error}")

    st.stop()





if not models:

    st.error("Aucun modèle n'a été chargé.")

    st.stop()









st.sidebar.title("⚙️ Paramètres")



page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Comparaison des modèles",

        "Matrices de confusion",

        "Courbe ROC",

        "Courbe Precision-Recall",

        "Analyse du modèle hybride"

    ]

)



st.sidebar.divider()



dataset_choice = st.sidebar.selectbox(

    "Jeu de données",

    ["Test", "Validation"]

)



if dataset_choice == "Test":

    X_original = data["X_test"]

    y_original = data["y_test"]

else:

    X_original = data["X_val"]

    y_original = data["y_val"]



st.sidebar.caption(

    f"Nombre total de transactions : {len(y_original):,}"

)



max_samples = st.sidebar.slider(

    "Nombre de transactions à analyser",

    min_value=1000,

    max_value=min(100000, len(y_original)),

    value=min(30000, len(y_original)),

    step=1000

)



X_current, y_current = sample_dataset(

    X_original,

    y_original,

    max_samples

)



st.sidebar.caption(

    f"Échantillon utilisé : {len(y_current):,}"

)



st.sidebar.caption(

    f"Modèles chargés : {len(models)}/{len(MODEL_FILES)}"

)








@st.cache_data(show_spinner="Calcul des scores des modèles...")

def cached_calculate_scores(X, model_items):

    

    return {}





with st.spinner("Évaluation des modèles en cours..."):



    scores_dict, scoring_errors = calculate_model_scores(

        models,

        X_current

    )



metrics = {}



for name, scores in scores_dict.items():



    if name == "Modèle hybride":

        threshold = get_threshold(models[name], default=0.5)



    elif name in UNSUPERVISED_MODELS:

        threshold = get_threshold(models[name], default=0.0)



    else:

        threshold = get_threshold(models[name], default=0.5)



    metrics[name] = calculate_metrics(

        y_current,

        scores,

        threshold

    )



metrics_df = pd.DataFrame(metrics).T



if not metrics_df.empty:

    metrics_df.index.name = "Modèle"



if scoring_errors:

    with st.expander("Voir les erreurs de calcul"):

        for name, error in scoring_errors.items():

            st.warning(f"{name} : {error}")









def show_section_title(title, description=None):

    st.markdown(

        f'<p class="section-title">{title}</p>',

        unsafe_allow_html=True

    )



    if description:

        st.caption(description)





def display_confusion_matrix(y_true, scores, threshold, title):

    y_pred = (scores >= threshold).astype(int)



    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]

    )



    fig = px.imshow(

        cm,

        text_auto=True,

        labels={

            "x": "Classe prédite",

            "y": "Classe réelle",

            "color": "Nombre"

        },

        x=["Normale", "Fraude"],

        y=["Normale", "Fraude"],

        color_continuous_scale="Blues",

        title=title

    )



    fig.update_layout(

        height=400,

        title_x=0.5

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )





def build_comparison_table(metrics_dataframe):

    columns = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-score",

        "ROC-AUC",

        "PR-AUC",

        "Alert Rate"

    ]



    available = [

        col for col in columns

        if col in metrics_dataframe.columns

    ]



    return metrics_dataframe[available].copy()





def show_metric_cards(row):

    columns = st.columns(4)



    values = [

        ("Accuracy", row["Accuracy"]),

        ("Precision", row["Precision"]),

        ("Recall", row["Recall"]),

        ("F1-score", row["F1-score"])

    ]



    for col, (label, value) in zip(columns, values):

        col.metric(

            label,

            f"{value:.4f}"

        )





# ============================================================

# 13. PAGE DASHBOARD

# ============================================================



if page == "Dashboard":



    show_section_title(

        "Vue générale",

        "Résumé du jeu de données et des modèles disponibles."

    )



    total_transactions = len(y_current)

    total_fraud = int(np.sum(y_current == 1))

    total_normal = int(np.sum(y_current == 0))

    fraud_percentage = (

        total_fraud / total_transactions * 100

        if total_transactions else 0

    )



    col1, col2, col3, col4 = st.columns(4)



    col1.metric(

        "Transactions analysées",

        f"{total_transactions:,}"

    )



    col2.metric(

        "Transactions normales",

        f"{total_normal:,}"

    )



    col3.metric(

        "Fraudes",

        f"{total_fraud:,}"

    )



    col4.metric(

        "Taux de fraude",

        f"{fraud_percentage:.4f}%"

    )



    st.divider()



    col1, col2 = st.columns(2)



    with col1:

        show_section_title("Distribution des classes")



        distribution = pd.DataFrame({

            "Classe": ["Normales", "Fraudes"],

            "Nombre": [total_normal, total_fraud]

        })



        fig = px.pie(

            distribution,

            names="Classe",

            values="Nombre",

            hole=0.55,

            title="Répartition des transactions"

        )



        st.plotly_chart(

            fig,

            use_container_width=True

        )



    with col2:

        show_section_title("Répartition des modèles")



        categories = pd.DataFrame({

            "Catégorie": [

                "Non supervisés",

                "Supervisés",

                "Hybride"

            ],

            "Nombre": [

                sum(n in models for n in UNSUPERVISED_MODELS),

                sum(n in models for n in SUPERVISED_MODELS),

                int("Modèle hybride" in models)

            ]

        })



        fig = px.bar(

            categories,

            x="Catégorie",

            y="Nombre",

            text="Nombre",

            title="Nombre de modèles chargés"

        )



        st.plotly_chart(

            fig,

            use_container_width=True

        )



    st.divider()



    show_section_title("Modèles disponibles")



    model_status = pd.DataFrame([

        {

            "Modèle": name,

            "Catégorie": (

                "Non supervisé"

                if name in UNSUPERVISED_MODELS

                else "Supervisé"

                if name in SUPERVISED_MODELS

                else "Hybride"

            ),

            "Statut": "Chargé" if name in models else "Absent"

        }

        for name in MODEL_FILES

    ])



    st.dataframe(

        model_status,

        use_container_width=True,

        hide_index=True

    )









elif page == "Comparaison des modèles":



    show_section_title(

        "Comparaison des performances",

        "Compare les modèles selon les principales métriques."

    )



    if metrics_df.empty:

        st.warning("Aucune métrique disponible.")

        st.stop()



    category = st.selectbox(

        "Catégorie des modèles",

        [

            "Tous les modèles",

            "Non supervisés",

            "Supervisés",

            "Hybride"

        ]

    )



    if category == "Non supervisés":

        selected_names = [

            n for n in metrics_df.index

            if n in UNSUPERVISED_MODELS

        ]



    elif category == "Supervisés":

        selected_names = [

            n for n in metrics_df.index

            if n in SUPERVISED_MODELS

        ]



    elif category == "Hybride":

        selected_names = [

            n for n in metrics_df.index

            if n == "Modèle hybride"

        ]



    else:

        selected_names = list(metrics_df.index)



    comparison = metrics_df.loc[selected_names]



    st.dataframe(

        build_comparison_table(comparison).style.format("{:.4f}"),

        use_container_width=True

    )



    st.divider()



    metric_choice = st.selectbox(

        "Métrique à comparer",

        [

            "Precision",

            "Recall",

            "F1-score",

            "PR-AUC",

            "ROC-AUC",

            "Accuracy",

            "Alert Rate"

        ]

    )



    plot_data = comparison.reset_index()



    fig = px.bar(

        plot_data,

        x="Modèle",

        y=metric_choice,

        color="Modèle",

        text_auto=".4f",

        title=f"Comparaison selon {metric_choice}"

    )



    fig.update_layout(

        showlegend=False,

        height=500,

        yaxis_range=[0, 1]

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )



    st.divider()



    show_section_title("Classement des modèles")



    ranking = comparison.sort_values(

        by="F1-score",

        ascending=False

    )



    ranking = ranking.reset_index()



    ranking.insert(

        0,

        "Rang",

        range(1, len(ranking) + 1)

    )



    st.dataframe(

        ranking[

            [

                "Rang",

                "Modèle",

                "Precision",

                "Recall",

                "F1-score",

                "PR-AUC"

            ]

        ].style.format({

            "Precision": "{:.4f}",

            "Recall": "{:.4f}",

            "F1-score": "{:.4f}",

            "PR-AUC": "{:.4f}"

        }),

        use_container_width=True,

        hide_index=True

    )









elif page == "Matrices de confusion":



    show_section_title(

        "Matrices de confusion",

        "Analyse des vrais positifs, faux positifs, vrais négatifs et faux négatifs."

    )



    available_models = list(scores_dict.keys())



    if not available_models:

        st.warning("Aucun modèle ne possède de score exploitable.")

        st.stop()



    selected_model = st.selectbox(

        "Sélectionner un modèle",

        available_models

    )



    artifact = models[selected_model]



    if selected_model in UNSUPERVISED_MODELS:

        threshold = get_threshold(artifact, default=0.0)

    else:

        threshold = get_threshold(artifact, default=0.5)



    st.info(f"Seuil utilisé : {threshold:.6f}")



    display_confusion_matrix(

        y_current,

        scores_dict[selected_model],

        threshold,

        f"Matrice de confusion — {selected_model}"

    )



    row = metrics[selected_model]



    col1, col2, col3, col4 = st.columns(4)



    col1.metric("TP", f"{row['TP']:,}")

    col2.metric("FP", f"{row['FP']:,}")

    col3.metric("TN", f"{row['TN']:,}")

    col4.metric("FN", f"{row['FN']:,}")



    st.divider()



    st.markdown("""

    **Interprétation :**



    - **TP** : fraude correctement détectée.

    - **FP** : transaction normale signalée comme fraude.

    - **TN** : transaction normale correctement identifiée.

    - **FN** : fraude non détectée.

    """)








elif page == "Courbe ROC":



    show_section_title(

        "Courbes ROC",

        "Compare la capacité des modèles à distinguer les transactions normales et frauduleuses."

    )



    fig = go.Figure()



    for name, scores in scores_dict.items():



        if len(np.unique(y_current)) < 2:

            continue



        fpr, tpr, _ = roc_curve(

            y_current,

            scores

        )



        auc_value = roc_auc_score(

            y_current,

            scores

        )



        fig.add_trace(

            go.Scatter(

                x=fpr,

                y=tpr,

                mode="lines",

                name=f"{name} (AUC = {auc_value:.4f})"

            )

        )



    fig.add_trace(

        go.Scatter(

            x=[0, 1],

            y=[0, 1],

            mode="lines",

            name="Classificateur aléatoire",

            line=dict(dash="dash")

        )

    )



    fig.update_layout(

        title="Comparaison des courbes ROC",

        xaxis_title="False Positive Rate",

        yaxis_title="True Positive Rate",

        height=600,

        xaxis=dict(range=[0, 1]),

        yaxis=dict(range=[0, 1])

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )









elif page == "Courbe Precision-Recall":



    show_section_title(

        "Courbes Precision-Recall",

        "Cette courbe est particulièrement utile pour un jeu de données fortement déséquilibré."

    )



    fig = go.Figure()



    for name, scores in scores_dict.items():



        precision_values, recall_values, _ = (

            precision_recall_curve(

                y_current,

                scores

            )

        )



        ap = average_precision_score(

            y_current,

            scores

        )



        fig.add_trace(

            go.Scatter(

                x=recall_values,

                y=precision_values,

                mode="lines",

                name=f"{name} (AP = {ap:.4f})"

            )

        )



    fig.update_layout(

        title="Comparaison des courbes Precision-Recall",

        xaxis_title="Recall",

        yaxis_title="Precision",

        height=600,

        xaxis=dict(range=[0, 1]),

        yaxis=dict(range=[0, 1])

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )









elif page == "Analyse du modèle hybride":



    show_section_title(

        "Analyse du modèle hybride",

        "Combinaison des scores du meilleur modèle non supervisé "

        "et du meilleur modèle supervisé."

    )



    if "Modèle hybride" not in models:

        st.warning("Le modèle hybride n'a pas été chargé.")

        st.stop()



    hybrid_artifact = models["Modèle hybride"]



    if not isinstance(hybrid_artifact, dict):

        st.error("Le fichier hybride ne contient pas le dictionnaire attendu.")

        st.stop()



    unsup_name = hybrid_artifact.get(

        "unsupervised_name",

        "Non spécifié"

    )



    sup_name = hybrid_artifact.get(

        "supervised_name",

        "Non spécifié"

    )



    experiment = hybrid_artifact.get(

        "supervised_experiment",

        "Non spécifiée"

    )



    alpha = float(hybrid_artifact.get("alpha", 0.5))



    threshold = float(

        hybrid_artifact.get("threshold", 0.5)

    )



    col1, col2, col3 = st.columns(3)



    col1.metric(

        "Modèle non supervisé",

        unsup_name

    )



    col2.metric(

        "Modèle supervisé",

        sup_name

    )



    col3.metric(

        "Expérience supervisée",

        str(experiment)

    )



    st.divider()



    show_section_title("Paramètres de combinaison")



    col1, col2 = st.columns(2)



    col1.metric(

        "Alpha",

        f"{alpha:.2f}"

    )



    col2.metric(

        "Seuil de décision",

        f"{threshold:.6f}"

    )



    st.markdown("**Formule utilisée :**")



    st.latex(
        r"S_{hybride} = "
        r"\\alpha S_{non\ supervisé} "
        r"+ (1-\\alpha) S_{supervisé}"
    )



    st.caption(

        "Les deux scores sont normalisés par rang percentile "

        "à partir de leurs distributions de référence."

    )



    st.divider()



    show_section_title("Performances du modèle hybride")



    hybrid_row = metrics["Modèle hybride"]



    show_metric_cards(hybrid_row)



    col1, col2 = st.columns(2)



    with col1:

        st.metric(

            "PR-AUC",

            f"{hybrid_row['PR-AUC']:.4f}"

        )



    with col2:

        st.metric(

            "ROC-AUC",

            f"{hybrid_row['ROC-AUC']:.4f}"

        )



    display_confusion_matrix(

        y_current,

        scores_dict["Modèle hybride"],

        threshold,

        "Matrice de confusion — Modèle hybride"

    )



    st.divider()



    show_section_title("Comparaison avec les autres modèles")



    selected_for_comparison = [

        name for name in [

            unsup_name,

            sup_name,

            "Modèle hybride"

        ]

        if name in metrics_df.index

    ]



    if selected_for_comparison:

        hybrid_comparison = metrics_df.loc[

            selected_for_comparison

        ]



        st.dataframe(

            build_comparison_table(

                hybrid_comparison

            ).style.format("{:.4f}"),

            use_container_width=True

        )








st.divider()



st.caption(

    "Projet de détection de fraude bancaire | "

    "Machine Learning | Modèles supervisés, non supervisés et hybride"

)
