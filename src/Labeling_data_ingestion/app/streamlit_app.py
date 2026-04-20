import sys
import time
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent
_SRC          = _PROJECT_ROOT / "src"

for _p in [str(_PROJECT_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from Labeling_data_ingestion.config import DATA_PATH, ThresholdConfig, TO_FILTER_PATH
    from Labeling_data_ingestion.data_handler.process_data import (
        build_dataset, load_data, build_prediction_dataset
    )
    from Labeling_data_ingestion.models.sklearn_models.sk_models import logistic_model
    from Labeling_data_ingestion.train.train import (
        train_test, train_tfidf, train_transformer, evaluate, save_model
    )
    _cfg                 = ThresholdConfig()
    _DEFAULT_DATA_PATH   = str(DATA_PATH)
    _DEFAULT_FILTER_PATH = str(TO_FILTER_PATH)
    _DEFAULT_HIGH_POS    = _cfg.high_pos
    _DEFAULT_HIGH_NEG    = _cfg.high_neg
    IMPORTS_OK           = True
    IMPORT_ERROR         = ""
except Exception as exc:
    IMPORTS_OK           = False
    IMPORT_ERROR         = str(exc)
    _DEFAULT_DATA_PATH   = "data/train_data.csv"
    _DEFAULT_FILTER_PATH = "data/to_filter.csv"
    _DEFAULT_HIGH_POS    = 0.70
    _DEFAULT_HIGH_NEG    = 0.30

st.set_page_config(
    page_title="Labeling Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

for k, v in dict(
    raw_df=None, dataset_df=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    pipe=None, eval_results=None,
    prediction_df=None, ranked_df=None,
    logs=[],
).items():
    if k not in st.session_state:
        st.session_state[k] = v


def _log(msg):
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, val) in zip(cols, metrics.items()):
        col.metric(label, val)


def _df_inspector(df: pd.DataFrame, key_prefix: str, default_rows: int = 20):
    all_cols = df.columns.tolist()
    chosen = st.multiselect("Columns", all_cols, default=all_cols, key=f"{key_prefix}_cols")
    n = st.slider("Rows", 5, min(500, len(df)), default_rows, key=f"{key_prefix}_rows")
    st.dataframe(df[chosen].head(n), use_container_width=True, height=320)


with st.sidebar:
    st.title("⚙️ Config")
    if not IMPORTS_OK:
        st.error(f"Import error:\n\n`{IMPORT_ERROR}`")
    else:
        st.success("✅ Modules loaded")

    st.divider()
    st.subheader("Paths")
    data_path       = st.text_input("Training CSV",         value=_DEFAULT_DATA_PATH)
    predict_in_path = st.text_input("Prediction input CSV", value="data/predict_data/unlabeled.csv")
    model_dir       = st.text_input("Trained models dir",   value="trained_models/")

    st.divider()
    st.subheader("Training")
    model_name = st.selectbox("Model", ["transformer", "tfidf"])
    test_size  = st.slider("Test size", 0.05, 0.40, 0.20, 0.05)
    seed       = st.number_input("Random seed", value=40, step=1)

    st.divider()
    st.subheader("Evaluation")
    eval_threshold = st.slider("Eval threshold", 0.01, 0.99, 0.26, 0.01)

    st.divider()
    st.subheader("Prediction thresholds")
    high_pos = st.slider("Certain positive  ≥", 0.01, 0.99, _DEFAULT_HIGH_POS, 0.01)
    high_neg = st.slider("Certain negative  ≤", 0.01, 0.99, _DEFAULT_HIGH_NEG, 0.01)
    if high_neg >= high_pos:
        st.warning("high_neg should be < high_pos")


tabs = st.tabs([
    "📥 1 · Data Ingestion",
    "🏋️ 2 · Training",
    "📊 3 · Evaluation",
    "🔮 4 · Prediction",
    "🏆 5 · MS MARCO",
    "📋 Logs",
])


with tabs[0]:
    st.header("Data Ingestion & Preprocessing")

    if st.button("📂 Load & Build Dataset", type="primary", key="btn_load"):
        if not IMPORTS_OK:
            st.error("Fix import errors first (see sidebar).")
        else:
            with st.spinner("Loading…"):
                try:
                    raw   = load_data(data_path)
                    built = build_dataset(raw)
                    st.session_state.raw_df     = raw
                    st.session_state.dataset_df = built
                    _log(f"Loaded {len(raw)} rows → built dataset {len(built)} rows")
                    st.success(f"Loaded **{len(raw)}** rows → **{len(built)}** after `build_dataset`")
                except Exception as exc:
                    st.error(str(exc)); _log(f"ERROR load: {exc}")

    if st.session_state.raw_df is not None:
        raw = st.session_state.raw_df
        st.subheader("Raw CSV")
        _metric_row({"Rows": len(raw), "Columns": len(raw.columns)})
        with st.expander("🔍 Inspect raw data", expanded=True):
            _df_inspector(raw, "raw")

    if st.session_state.dataset_df is not None:
        built = st.session_state.dataset_df
        st.subheader("Built Dataset")
        vc = built["relevant"].value_counts()
        _metric_row({
            "Total rows":     len(built),
            "Relevant (1)":   int(vc.get(1.0, 0)),
            "Irrelevant (0)": int(vc.get(0.0, 0)),
            "Null relevant":  int(built["relevant"].isnull().sum()),
        })

        col_a, col_b = st.columns([3, 2])
        with col_a:
            with st.expander("🔍 Inspect built dataset", expanded=True):
                rel_f = st.radio("Filter", ["All", "Relevant (1)", "Irrelevant (0)"],
                                 horizontal=True, key="built_filter")
                df_show = built.copy()
                if rel_f == "Relevant (1)":     df_show = df_show[df_show["relevant"] == 1.0]
                elif rel_f == "Irrelevant (0)": df_show = df_show[df_show["relevant"] == 0.0]
                _df_inspector(df_show, "built")
        with col_b:
            with st.expander("📊 Class balance", expanded=True):
                fig, ax = plt.subplots(figsize=(4, 3))
                vc.plot(kind="bar", ax=ax, color=["#ef476f", "#06d6a0"], edgecolor="white")
                ax.set_xlabel("relevant"); ax.set_ylabel("count")
                ax.set_xticklabels(["0", "1"], rotation=0)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)


with tabs[1]:
    st.header("Model Training")

    if st.session_state.dataset_df is None:
        st.info("Complete **Tab 1** first.")
    else:
        save_path = f"{model_dir.rstrip('/')}/{model_name}.rfk"

        if st.button(f"🏋️ Train {model_name}", type="primary", key="btn_train"):
            with st.spinner(f"Splitting and training {model_name}…"):
                try:
                    X_tr, X_te, y_tr, y_te = train_test(
                        st.session_state.dataset_df, float(test_size), int(seed)
                    )
                    st.session_state.X_train = X_tr
                    st.session_state.X_test  = X_te
                    st.session_state.y_train = y_tr
                    st.session_state.y_test  = y_te
                    _log(f"Split: {len(X_tr)} train / {len(X_te)} test")

                    base = logistic_model()
                    fn   = {"tfidf": train_tfidf, "transformer": train_transformer}[model_name]
                    pipe = fn(base, X_tr, y_tr)
                    save_model(pipe, save_path)
                    st.session_state.pipe = pipe
                    _log(f"Trained {model_name}, saved → {save_path}")
                    st.success(
                        f"Split: **{len(X_tr)}** train / **{len(X_te)}** test &nbsp;|&nbsp; "
                        f"Model saved to `{save_path}`"
                    )
                except Exception as exc:
                    st.error(str(exc)); _log(f"ERROR train: {exc}")

        st.divider()
        st.subheader("Load existing model")
        load_path = st.text_input("Model path (.rfk)", value=save_path, key="load_path_input")
        if st.button("📂 Load model", key="btn_load_model"):
            try:
                pipe = joblib.load(load_path)
                st.session_state.pipe = pipe
                _log(f"Loaded model from {load_path}")
                st.success(f"✅ Loaded `{load_path}`")
            except Exception as exc:
                st.error(str(exc))

        if st.session_state.pipe is not None:
            st.success("🟢 Model in memory")
            with st.expander("Pipeline steps"):
                for name, step in st.session_state.pipe.steps:
                    st.markdown(f"- **{name}** → `{type(step).__name__}`")


with tabs[2]:
    st.header("Model Evaluation")

    if st.session_state.pipe is None:
        st.info("Train or load a model in **Tab 2** first.")
    elif st.session_state.X_test is None:
        st.info("Run training first so the test split is available.")
    else:
        if st.button("📊 Evaluate", type="primary", key="btn_eval"):
            with st.spinner("Evaluating…"):
                try:
                    results = evaluate(
                        st.session_state.pipe,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        eval_threshold,
                    )
                    st.session_state.eval_results = results
                    _log(f"Eval t={eval_threshold:.2f} acc={results['accuracy']:.3f} f1={results['f1']:.3f} roc={results['roc_auc']:.3f}")
                except Exception as exc:
                    st.error(str(exc)); _log(f"ERROR eval: {exc}")

        if st.session_state.eval_results is not None:
            r = st.session_state.eval_results
            _metric_row({
                "Accuracy":  f"{r['accuracy']:.3f}",
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1":        f"{r['f1']:.3f}",
                "ROC AUC":   f"{r['roc_auc']:.3f}",
            })
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
                ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            with c2:
                st.subheader("Classification Report")
                st.code(r["report"], language=None)


with tabs[3]:
    st.header("Prediction / Inference")

    pred_model_path = st.text_input("Model path", value=f"{model_dir.rstrip('/')}/{model_name}.rfk", key="pred_model_path")
    c1, c2 = st.columns(2)
    save_certain = c1.text_input("Save certain labels →", value="data/certain_auto/auto_labeled.csv")
    save_manual  = c2.text_input("Save manual review →",  value="data/manual_review/manual_review.csv")

    if st.button("🔮 Run Prediction", type="primary", key="btn_predict"):
        if not IMPORTS_OK:
            st.error("Fix imports first.")
        else:
            with st.spinner("Running inference…"):
                try:
                    df_p   = load_data(predict_in_path)
                    df_p   = df_p.dropna(subset=["theme"]).copy()
                    pipe   = joblib.load(pred_model_path)
                    X      = build_prediction_dataset(df_p)
                    scores = pd.Series(pipe.predict_proba(X)[:, 1], index=df_p.index)

                    mask_pos = scores >= high_pos
                    mask_neg = scores <= high_neg
                    mask_man = ~mask_pos & ~mask_neg

                    df_p.loc[mask_pos, "relevant"] = 1.0
                    df_p.loc[mask_neg, "relevant"] = 0.0
                    df_p.loc[mask_man, "relevant"] = None
                    df_p["score"] = scores

                    certain_df = df_p.loc[mask_pos | mask_neg]
                    manual_df  = df_p.loc[mask_man]
                    Path(save_certain).parent.mkdir(parents=True, exist_ok=True)
                    Path(save_manual).parent.mkdir(parents=True, exist_ok=True)
                    certain_df.to_csv(save_certain, mode="a", index=False, header=not Path(save_certain).exists())
                    manual_df.to_csv(save_manual,   mode="a", index=False, header=not Path(save_manual).exists())

                    st.session_state.prediction_df = df_p
                    _log(f"Prediction: +{mask_pos.sum()} pos | -{mask_neg.sum()} neg | {mask_man.sum()} manual")
                    st.success("✅ Prediction complete")
                except Exception as exc:
                    st.error(str(exc)); _log(f"ERROR predict: {exc}")

    if st.session_state.prediction_df is not None:
        df_p   = st.session_state.prediction_df
        scores = df_p["score"]
        c_pos  = df_p[scores >= high_pos]
        c_neg  = df_p[scores <= high_neg]
        manual = df_p[(scores > high_neg) & (scores < high_pos)]

        _metric_row({
            "✅ Certain Relevant":   len(c_pos),
            "❌ Certain Irrelevant": len(c_neg),
            "🔍 Manual Review":     len(manual),
            "Total":                len(df_p),
        })

        with st.expander("📈 Score distribution", expanded=True):
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.hist(scores, bins=60, color="#4895ef", alpha=0.85, edgecolor="white")
            ax.axvline(high_pos, color="#06d6a0", lw=2, linestyle="--", label=f"high_pos = {high_pos}")
            ax.axvline(high_neg, color="#ef476f", lw=2, linestyle="--", label=f"high_neg = {high_neg}")
            ax.set_xlabel("Confidence score"); ax.set_ylabel("Count")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        subset = st.radio("Inspect subset", ["All", "Certain Relevant", "Certain Irrelevant", "Manual Review"], horizontal=True)
        df_show = {"All": df_p, "Certain Relevant": c_pos, "Certain Irrelevant": c_neg, "Manual Review": manual}[subset]
        if len(df_show):
            _df_inspector(df_show.sort_values("score", ascending=False), "pred")
        else:
            st.info("No rows in this subset.")


with tabs[4]:
    st.header("MS MARCO Cross-Encoder Re-Ranking")

    c1, c2 = st.columns(2)
    marco_in   = c1.text_input("Certain positives CSV", value="data/certain_auto/auto_labeled.csv", key="marco_in")
    ranked_out = c2.text_input("Ranked output path",    value="data/ranked/ranked_data.csv")

    if st.button("🏆 Run MS MARCO", type="primary", key="btn_marco"):
        with st.spinner("Loading cross-encoder… (first run downloads model)"):
            try:
                from sentence_transformers import CrossEncoder
                df_r = pd.read_csv(marco_in)
                df_r = df_r[df_r["relevant"] == 1.0].copy()
                if df_r.empty:
                    st.error("No certain positives (relevant == 1.0) found in the file.")
                else:
                    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
                    df_r["ranking"] = ce.predict(list(zip(df_r["theme"], df_r["title"])))
                    Path(ranked_out).parent.mkdir(parents=True, exist_ok=True)
                    df_r.to_csv(ranked_out, index=False)
                    st.session_state.ranked_df = df_r
                    _log(f"MS MARCO: {len(df_r)} certain positives ranked → {ranked_out}")
                    st.success(f"✅ Ranked **{len(df_r)}** rows — saved to `{ranked_out}`")
            except Exception as exc:
                st.error(str(exc)); _log(f"ERROR marco: {exc}")

    st.divider()
    st.subheader("Load existing ranked output")
    if st.button("📂 Load ranked CSV", key="btn_load_ranked"):
        try:
            st.session_state.ranked_df = pd.read_csv(ranked_out)
            st.success(f"✅ Loaded {len(st.session_state.ranked_df)} rows")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.ranked_df is not None:
        df_r = st.session_state.ranked_df
        lo   = float(df_r["ranking"].min())
        hi   = float(df_r["ranking"].max())

        _metric_row({
            "Total ranked": len(df_r),
            "Max score":    f"{hi:.3f}",
            "Min score":    f"{lo:.3f}",
            "Mean":         f"{df_r['ranking'].mean():.3f}",
        })

        with st.expander("📈 Ranking distribution", expanded=True):
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.hist(df_r["ranking"], bins=60, color="#7b2d8b", alpha=0.85, edgecolor="white")
            ax.set_xlabel("MS MARCO score"); ax.set_ylabel("Count")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.divider()
        st.subheader("Filter & Export")

        rank_thresh = st.number_input(
            "Keep rows with ranking ≥",
            value=round(lo + (hi - lo) * 0.25, 3),
            step=0.1,
            format="%.3f",
            key="rank_thresh",
        )

        filtered = df_r[df_r["ranking"] >= rank_thresh].sort_values("ranking", ascending=False)
        kept_pct = len(filtered) / len(df_r) * 100 if len(df_r) else 0

        _metric_row({
            "Kept":      len(filtered),
            "Dropped":   len(df_r) - len(filtered),
            "% kept":    f"{kept_pct:.1f}%",
            "Threshold": f"{rank_thresh:.3f}",
        })

        save_filtered = st.text_input("Save filtered output →", value="data/ranked/filtered_for_enrichment.csv")

        if st.button("💾 Save filtered rows", type="primary", key="btn_save_filtered"):
            if filtered.empty:
                st.warning("Nothing to save at this threshold.")
            else:
                try:
                    Path(save_filtered).parent.mkdir(parents=True, exist_ok=True)
                    filtered.to_csv(save_filtered, index=False)
                    _log(f"Saved {len(filtered)} filtered rows (≥{rank_thresh:.3f}) → {save_filtered}")
                    st.success(f"✅ Saved **{len(filtered)}** rows to `{save_filtered}`")
                except Exception as exc:
                    st.error(str(exc))

        st.divider()
        _df_inspector(filtered, "ranked")


with tabs[5]:
    st.header("Run Logs")
    if st.button("🗑️ Clear", key="btn_clear_logs"):
        st.session_state.logs = []
    if st.session_state.logs:
        st.code("\n".join(reversed(st.session_state.logs)), language=None)
    else:
        st.info("No log entries yet.")