
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
POPULARITY_THRESHOLD = 70
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "spotify_tracks_dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Os atributos musicais do arquivo oficial. Identificadores, artista, álbum e nome
# da faixa são metadados e não entram no modelo para evitar memorização.
NUMERIC_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]
CATEGORICAL_FEATURES = ["explicit", "track_genre"]
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["popularity"]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATA_PATH}. "
        )

    df = pd.read_csv(DATA_PATH)
    index_columns = [column for column in df.columns if column.startswith("Unnamed:")]
    df = df.drop(columns=index_columns, errors="ignore")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing}")

    for column in NUMERIC_FEATURES + ["popularity"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Mantém explicit como categoria legível e consistente após a leitura do CSV.
    df["explicit"] = (
        df["explicit"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": "Sim", "false": "Não"})
        .fillna("Não informado")
    )
    df["track_genre"] = df["track_genre"].astype(str).str.strip()
    df = df.dropna(subset=REQUIRED_COLUMNS).drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError("Não restaram registros válidos após a limpeza do dataset.")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Cria a classe alvo binária e separa atributos de entrada e rótulo."""
    prepared = df.copy()
    prepared["popularidade_alta"] = (
        prepared["popularity"] >= POPULARITY_THRESHOLD
    ).astype(int)
    x = prepared[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = prepared["popularidade_alta"]

    if y.nunique() < 2:
        raise ValueError("A variável alvo possui apenas uma classe após a preparação.")
    return x, y


def build_preprocessor() -> ColumnTransformer:
    """Padroniza números e codifica as categorias do Spotify."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> Dict[str, Pipeline]:
    """Monta os modelos solicitados no trabalho: árvore e MLP."""
    return {
        "arvore_decisao": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "rede_neural_mlp": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(48, 24),
                        activation="relu",
                        solver="adam",
                        max_iter=300,
                        batch_size=256,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def get_feature_names(model: Pipeline) -> List[str]:
    """Recupera os nomes após o pré-processamento para explicar a árvore."""
    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]
    return list(NUMERIC_FEATURES) + encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()


def evaluate_model(
    name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calcula métricas e salva a matriz de confusão de um modelo."""
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "modelo": name,
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "verdadeiro_negativo": tn,
        "falso_positivo": fp,
        "falso_negativo": fn,
        "verdadeiro_positivo": tp,
    }

    print("\n" + "=" * 70)
    print(f"Relatório de classificação - {name}")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["Não popular", "Popular"], zero_division=0))

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Não popular", "Popular"])
    display.plot(values_format="d")
    plt.title(f"Matriz de confusão - {name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"matriz_confusao_{name}.png", dpi=180)
    plt.close()
    return metrics


def plot_dataset_charts(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["popularity"], bins=21, color="#1DB954", edgecolor="white")
    plt.title("Distribuição da popularidade das faixas")
    plt.xlabel("Popularidade")
    plt.ylabel("Quantidade de faixas")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribuicao_popularidade.png", dpi=180)
    plt.close()

    labels = (df["popularity"] >= POPULARITY_THRESHOLD).map({False: "Não popular", True: "Popular"})
    counts = labels.value_counts().reindex(["Não popular", "Popular"], fill_value=0)
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values, color=["#64748B", "#1DB954"])
    plt.title("Distribuição das classes")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de faixas")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribuicao_classes.png", dpi=180)
    plt.close()


def plot_comparison(metrics_df: pd.DataFrame) -> None:
    metric_columns = ["acuracia", "precisao", "recall", "f1_score"]
    ax = metrics_df.set_index("modelo")[metric_columns].plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    ax.set_title("Comparação de desempenho entre modelos")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor da métrica")
    ax.legend(title="Métrica", loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparacao_modelos.png", dpi=180)
    plt.close()


def plot_decision_tree(model: Pipeline) -> None:
    classifier = model.named_steps["classifier"]
    plt.figure(figsize=(22, 11))
    plot_tree(
        classifier,
        max_depth=3,
        feature_names=get_feature_names(model),
        class_names=["Não popular", "Popular"],
        filled=True,
        fontsize=7,
    )
    plt.title("Visualização parcial da Árvore de Decisão")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "arvore_decisao_visualizacao.png", dpi=180)
    plt.close()


def plot_feature_importance(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    importance_df = pd.DataFrame(
        {"atributo": get_feature_names(model), "importancia": classifier.feature_importances_}
    ).sort_values("importancia", ascending=False, ignore_index=True)
    importance_df.to_csv(RESULTS_DIR / "importancia_atributos_arvore_decisao.csv", index=False)

    top = importance_df.head(top_n).sort_values("importancia")
    plt.figure(figsize=(9, 7))
    plt.barh(top["atributo"], top["importancia"], color="#1DB954")
    plt.title("Atributos mais importantes - Árvore de Decisão")
    plt.xlabel("Importância")
    plt.ylabel("Atributo")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "importancia_atributos_arvore_decisao.png", dpi=180)
    plt.close()
    return importance_df


def plot_mlp_training_curves(model: Pipeline) -> None:
    classifier = model.named_steps["classifier"]
    loss_curve = getattr(classifier, "loss_curve_", [])
    validation_scores = getattr(classifier, "validation_scores_", [])
    if len(loss_curve) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(loss_curve) + 1), loss_curve, color="#1DB954")
    axes[0].set(title="Perda durante o treinamento", xlabel="Época", ylabel="Loss")
    if len(validation_scores) > 0:
        axes[1].plot(range(1, len(validation_scores) + 1), validation_scores, color="#2563EB")
        axes[1].set(title="Acurácia na validação", xlabel="Época", ylabel="Score", ylim=(0, 1))
    else:
        axes[1].axis("off")
    fig.suptitle("Curva de treinamento - Rede Neural MLP")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "curva_treinamento_rede_neural_mlp.png", dpi=180)
    plt.close(fig)


def write_execution_summary(
    metrics_df: pd.DataFrame, feature_importance_df: pd.DataFrame, row_count: int
) -> None:
    best = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]
    lines = [
        "# Resumo da Execução",
        "",
        "## Dataset utilizado",
        "",
        "- Fonte: Spotify Tracks Dataset.",
        f"- Registros válidos após limpeza: {row_count:,}.",
        f"- Classe positiva: popularity >= {POPULARITY_THRESHOLD}.",
        "",
        "## Resultado principal",
        "",
        f"O melhor modelo pelo F1-score foi `{best['modelo']}` com F1-score de {best['f1_score']:.4f}.",
        "",
        "| Modelo | Acurácia | Precisão | Recall | F1-score | VP | VN | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            "| {modelo} | {acuracia:.4f} | {precisao:.4f} | {recall:.4f} | {f1_score:.4f} | "
            "{verdadeiro_positivo:.0f} | {verdadeiro_negativo:.0f} | {falso_positivo:.0f} | "
            "{falso_negativo:.0f} |".format(**row.to_dict())
        )
    lines.extend(["", "## Atributos mais importantes na Árvore de Decisão", "", "| Atributo | Importância |", "|---|---:|"])
    for _, row in feature_importance_df.head(5).iterrows():
        lines.append(f"| {row['atributo']} | {row['importancia']:.4f} |")
    (RESULTS_DIR / "resumo_execucao.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    x, y = prepare_data(df)

    print("=" * 70)
    print("Trabalho Final - Inteligência Artificial")
    print("Predição de popularidade com Spotify Tracks Dataset")
    print("=" * 70)
    print(f"Dataset carregado: {len(df):,} linhas e {len(df.columns)} colunas")
    print(f"Atributos de entrada: {len(NUMERIC_FEATURES)} numéricos e {len(CATEGORICAL_FEATURES)} categóricos")
    print("\nDistribuição da variável alvo:")
    print(y.value_counts().rename(index={0: "Não popular", 1: "Popular"}))
    plot_dataset_charts(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nDivisão: {len(x_train):,} para treino e {len(x_test):,} para teste")

    metrics_list = []
    models = build_models()
    for name, model in models.items():
        print("\n" + "-" * 70)
        print(f"Treinando modelo: {name}")
        model.fit(x_train, y_train)
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        metrics_list.append(evaluate_model(name, model, x_test, y_test))

    plot_decision_tree(models["arvore_decisao"])
    feature_importance_df = plot_feature_importance(models["arvore_decisao"])
    plot_mlp_training_curves(models["rede_neural_mlp"])
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(RESULTS_DIR / "metricas_modelos.csv", index=False)
    plot_comparison(metrics_df)
    write_execution_summary(metrics_df, feature_importance_df, len(df))

    print("\nMétricas finais:")
    print(metrics_df.to_string(index=False))
    best = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]
    print(f"\nMelhor modelo pelo F1-score: {best['modelo']} ({best['f1_score']:.4f})")


if __name__ == "__main__":
    main()
