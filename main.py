"""
main.py — CLI entry point for the LSTM Short Text Classification project.

Usage
─────
  python main.py preprocess              # Preprocess raw dataset
  python main.py train lstm              # Train BiLSTM (proposed model)
  python main.py train dnn               # Train TF-IDF + DNN baseline
  python main.py train xlmr              # Fine-tune XLM-RoBERTa
  python main.py train                   # Train all three models
  python main.py ablation                # Ablation study (LSTM / BiLSTM / BiLSTM+Attn)
  python main.py evaluate                # Evaluate all models → comparison.json
  python main.py analyze [model]         # Error analysis (default: lstm)
  python main.py visualize               # Generate figures
  python main.py all                     # Full pipeline
  python main.py predict "input text"    # Quick prediction with BiLSTM
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _banner(text: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0].lower()

    # ── preprocess ────────────────────────────────────────────────────────────
    if cmd == "preprocess":
        _banner("PREPROCESSING")
        from src.preprocessing.preprocess import run_preprocessing
        run_preprocessing()

    # ── train ─────────────────────────────────────────────────────────────────
    elif cmd == "train":
        sub = args[1].lower() if len(args) > 1 else "all"

        if sub in ("all", "lstm"):
            _banner("TRAINING — BiLSTM (Proposed)")
            from src.training.train_lstm import train_lstm
            train_lstm()

        if sub in ("all", "dnn"):
            _banner("TRAINING — DNN + TF-IDF (Baseline)")
            from src.training.train_dnn import train_dnn
            train_dnn()

        if sub in ("all", "xlmr"):
            _banner("TRAINING — XLM-RoBERTa (Comparison)")
            from src.training.train_xlmr import train_xlmr
            train_xlmr()

        if sub not in ("all", "lstm", "dnn", "xlmr"):
            print(f"Unknown model '{sub}'. Choose: lstm | dnn | xlmr | (omit for all)")

    # ── ablation ──────────────────────────────────────────────────────────────
    elif cmd == "ablation":
        _banner("ABLATION STUDY")
        from src.training.train_ablation import run_ablation
        results = run_ablation()
        print("\nAblation Results:")
        print(f"  {'Variant':<35} {'Accuracy':>9} {'F1':>9} {'Params':>12}")
        print("  " + "-" * 70)
        for name, m in results.items():
            print(f"  {name:<35} {m['accuracy']:>9.4f} {m['f1']:>9.4f} {m['num_params']:>12,}")

    # ── analyze ───────────────────────────────────────────────────────────────
    elif cmd == "analyze":
        model_name = args[1].lower() if len(args) > 1 else "lstm"
        _banner(f"ERROR ANALYSIS — {model_name.upper()}")
        from src.evaluation.error_analysis import analyze_errors, print_error_summary
        report = analyze_errors(model_name)
        print_error_summary(report)

    # ── evaluate ──────────────────────────────────────────────────────────────
    elif cmd == "evaluate":
        _banner("EVALUATION")
        from src.evaluation.evaluate import run_evaluation
        run_evaluation()

    # ── visualize ─────────────────────────────────────────────────────────────
    elif cmd == "visualize":
        _banner("VISUALIZATION")
        from src.evaluation.visualization import generate_all_figures
        generate_all_figures()

    # ── predict ───────────────────────────────────────────────────────────────
    elif cmd == "predict":
        text = " ".join(args[1:]) if len(args) > 1 else ""
        if not text:
            print("Usage: python main.py predict <text>")
            return
        from src.inference.predict import LSTMPredictor
        predictor = LSTMPredictor()
        result    = predictor.predict(text)
        print(f"\nInput       : {text}")
        print(f"Emotion     : {result['predicted_name']} ({result['predicted_label']})")
        print(f"Confidence  : {result['confidence']:.2%}")
        print("Probabilities:")
        for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {label}: {prob:.2%}")

    # ── all ────────────────────────────────────────────────────────────────────
    elif cmd == "all":
        _banner("FULL PIPELINE")
        from src.preprocessing.preprocess    import run_preprocessing
        from src.training.train_dnn          import train_dnn
        from src.training.train_lstm         import train_lstm
        from src.training.train_xlmr         import train_xlmr
        from src.training.train_ablation     import run_ablation
        from src.evaluation.evaluate         import run_evaluation
        from src.evaluation.visualization    import generate_all_figures
        from src.evaluation.error_analysis   import analyze_errors

        run_preprocessing()
        train_lstm()
        train_dnn()
        train_xlmr()
        run_ablation()
        run_evaluation()
        generate_all_figures()
        analyze_errors("lstm")
        print("\n✅  Full pipeline complete!")

    else:
        print(f"Unknown command: '{cmd}'")
        print(__doc__)


if __name__ == "__main__":
    main()
