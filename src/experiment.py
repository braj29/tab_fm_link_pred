"""Shared experiment entry point for TabICL/TabPFN runs."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
import importlib.util
import sys

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_splits, prepare_data
from metrics import (
    binary_classification_metrics,
    filtered_ranking_metrics_binary,
    sampled_ranking_metrics_binary,
)

_model_path = SRC_DIR / "model.py"
_spec = importlib.util.spec_from_file_location("tab_fm_model", _model_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load model definitions from {_model_path}")
_model_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model_module)
build_limix = _model_module.build_limix
build_tabicl = _model_module.build_tabicl
build_tabpfn = _model_module.build_tabpfn
build_tabdpt = _model_module.build_tabdpt
build_saint = _model_module.build_saint
build_kgbert = _model_module.build_kgbert
build_rotatee = _model_module.build_rotatee
build_complex = _model_module.build_complex
build_tag = _model_module.build_tag
build_tag_graph = _model_module.build_tag_graph


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="tabicl",
        choices=[
            "tabicl",
            "tabpfn",
            "limix",
            "tabdpt",
            "saint",
            "kgbert",
            "rotatee",
            "complex",
            "complex-kge",
            "tag",
            "tag-graph",
        ],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional train split size cap (None keeps full FB15k-237 train set).",
    )
    parser.add_argument(
        "--max-valid",
        type=int,
        default=None,
        help="Optional validation split size cap.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional test split size cap.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap applied to train/valid/test splits (debug).",
    )
    parser.add_argument(
        "--overfit-small",
        action="store_true",
        help="Train on 200 triples and evaluate on the same set.",
    )
    parser.add_argument(
        "--n-neg-per-pos",
        type=int,
        default=5,
        help="Number of negatives to sample per positive triple for training.",
    )
    parser.add_argument(
        "--n-neg-eval",
        type=int,
        default=100,
        help="Negatives per positive for validation/test evaluation.",
    )
    parser.add_argument(
        "--no-filter-unseen",
        action="store_true",
        help="Keep validation/test triples with unseen entities.",
    )
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help="Sample relation-consistent negatives (harder).",
    )
    parser.add_argument(
        "--corrupt-head-prob",
        type=float,
        default=0.5,
        help="Probability of corrupting head when sampling negatives.",
    )
    parser.add_argument(
        "--no-classification-metrics",
        action="store_false",
        dest="classification_metrics",
        help="Disable binary classification metrics (accuracy/F1/ROC-AUC).",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification metrics.",
    )
    parser.add_argument(
        "--ranking-batchsize",
        type=int,
        default=512,
        help="Batch size for ranking evaluation candidate scoring.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=2000,
        help="Max candidate entities per ranking query (None for full set).",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=42,
        help="RNG seed for candidate sampling.",
    )
    parser.add_argument(
        "--sampled-ranking-metrics",
        action="store_true",
        help="Compute ranking metrics by ranking positives among sampled negatives.",
    )
    parser.add_argument(
        "--sampled-ranking-negatives",
        type=int,
        default=None,
        help="Negatives per positive for sampled ranking (defaults to --n-neg-eval).",
    )
    parser.set_defaults(classification_metrics=True)
    parser.add_argument(
        "--limix-model-id",
        type=str,
        default="stableai-org/LimiX-16M",
        help="Hugging Face repo id for LimiX weights.",
    )
    parser.add_argument(
        "--limix-model-file",
        type=str,
        default="LimiX-16M.ckpt",
        help="Model filename within the LimiX repo.",
    )
    parser.add_argument(
        "--limix-model-path",
        type=str,
        default=None,
        help="Optional local path to LimiX checkpoint.",
    )
    parser.add_argument(
        "--limix-path",
        type=str,
        default=None,
        help="Path to local LimiX repo (adds to PYTHONPATH).",
    )
    parser.add_argument(
        "--limix-cache-dir",
        type=str,
        default="./cache",
        help="Cache dir for downloaded LimiX weights.",
    )
    parser.add_argument(
        "--limix-config",
        type=str,
        default="config/cls_default_noretrieval.json",
        help="Inference config path for LimiX.",
    )
    parser.add_argument(
        "--tabdpt-path",
        type=str,
        default=None,
        help="Path to local TabDPT inference repo (adds to PYTHONPATH).",
    )
    parser.add_argument(
        "--tabdpt-weights",
        type=str,
        default=None,
        help="Optional local path to TabDPT weights.",
    )
    parser.add_argument(
        "--tabdpt-n-ensembles",
        type=int,
        default=8,
        help="Number of TabDPT ensembles for inference.",
    )
    parser.add_argument(
        "--tabdpt-temperature",
        type=float,
        default=0.8,
        help="Softmax temperature for TabDPT.",
    )
    parser.add_argument(
        "--tabdpt-context-size",
        type=int,
        default=2048,
        help="Context size for TabDPT.",
    )
    parser.add_argument(
        "--tabdpt-permute-classes",
        action="store_true",
        help="Permute class labels across ensembles for TabDPT.",
    )
    parser.add_argument(
        "--saint-path",
        type=str,
        default=None,
        help="Path to local SAINT repo (adds to PYTHONPATH).",
    )
    parser.add_argument(
        "--saint-embedding-size",
        type=int,
        default=32,
        help="SAINT embedding size.",
    )
    parser.add_argument(
        "--saint-depth",
        type=int,
        default=6,
        help="SAINT transformer depth.",
    )
    parser.add_argument(
        "--saint-heads",
        type=int,
        default=8,
        help="SAINT attention heads.",
    )
    parser.add_argument(
        "--saint-attentiontype",
        type=str,
        default="colrow",
        help="SAINT attention type (col/row/colrow/justmlp/attn/attnmlp).",
    )
    parser.add_argument(
        "--saint-lr",
        type=float,
        default=1e-4,
        help="SAINT learning rate.",
    )
    parser.add_argument(
        "--saint-epochs",
        type=int,
        default=20,
        help="SAINT epochs.",
    )
    parser.add_argument(
        "--saint-batchsize",
        type=int,
        default=256,
        help="SAINT batch size.",
    )
    parser.add_argument(
        "--kgbert-model",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model id for KG-BERT baseline.",
    )
    parser.add_argument(
        "--kgbert-max-length",
        type=int,
        default=64,
        help="Max sequence length for KG-BERT tokenization.",
    )
    parser.add_argument(
        "--kgbert-lr",
        type=float,
        default=2e-5,
        help="KG-BERT learning rate.",
    )
    parser.add_argument(
        "--kgbert-epochs",
        type=int,
        default=3,
        help="KG-BERT fine-tuning epochs.",
    )
    parser.add_argument(
        "--kgbert-batchsize",
        type=int,
        default=16,
        help="KG-BERT batch size.",
    )
    parser.add_argument(
        "--rotatee-epochs",
        type=int,
        default=100,
        help="RotatE training epochs (PyKEEN).",
    )
    parser.add_argument(
        "--rotatee-dim",
        type=int,
        default=200,
        help="RotatE embedding dimension (PyKEEN).",
    )
    parser.add_argument(
        "--rotatee-batchsize",
        type=int,
        default=1024,
        help="RotatE batch size (PyKEEN).",
    )
    parser.add_argument(
        "--complex-epochs",
        type=int,
        default=100,
        help="ComplEx training epochs (PyKEEN).",
    )
    parser.add_argument(
        "--complex-lr",
        type=float,
        default=1e-3,
        help="ComplEx learning rate (PyKEEN).",
    )
    parser.add_argument(
        "--complex-dim",
        type=int,
        default=200,
        help="ComplEx embedding dimension (PyKEEN).",
    )
    parser.add_argument(
        "--complex-batchsize",
        type=int,
        default=1024,
        help="ComplEx batch size (PyKEEN).",
    )
    parser.add_argument(
        "--complex-eval-batchsize",
        type=int,
        default=512,
        help="ComplEx evaluation batch size (PyKEEN rank-based evaluator).",
    )
    parser.add_argument(
        "--complex-eval-slice-size",
        type=int,
        default=None,
        help="ComplEx evaluation slice size (PyKEEN rank-based evaluator).",
    )
    parser.add_argument(
        "--complex-eval-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for ComplEx evaluation (can use CPU to avoid CUDA allocator errors).",
    )
    parser.add_argument(
        "--tag-path",
        type=str,
        default=None,
        help="Path to local TAG repo (adds to PYTHONPATH).",
    )
    parser.add_argument(
        "--tag-base-model",
        type=str,
        default="tabicl",
        choices=["tabicl", "tabpfn"],
        help="Base model used inside TAG wrapper.",
    )
    parser.add_argument(
        "--tag-max-train-rows",
        type=int,
        default=10000,
        help="Max rows sampled by TAG wrapper for in-context training.",
    )
    parser.add_argument(
        "--tag-max-cells-per-batch",
        type=int,
        default=5_000_000,
        help="Max cells per inference batch for TAG wrapper.",
    )
    parser.add_argument(
        "--tag-graph-node-dim",
        type=int,
        default=64,
        help="Node feature dimension for TAG graph pipeline.",
    )
    parser.add_argument(
        "--tag-graph-relation-dim",
        type=int,
        default=16,
        help="Relation feature dimension for TAG graph pipeline.",
    )
    parser.add_argument(
        "--tag-graph-hops",
        type=int,
        default=4,
        help="Number of hops for TAG low-pass graph features.",
    )
    parser.add_argument(
        "--tag-graph-features",
        type=str,
        default="X,L1,L2,L3,L4,RW20,LE20",
        help="Comma-separated TAG graph features to include.",
    )
    parser.add_argument(
        "--tag-graph-gpse",
        action="store_true",
        help="Enable GPSE features for TAG graph pipeline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_metrics.json",
        help="Path to write metrics JSON.",
    )
    return parser.parse_args(argv)


def _metric_results_to_dict(results) -> dict:
    if hasattr(results, "to_dict"):
        return results.to_dict()
    return dict(results)


def _run_complex_kge(args: argparse.Namespace, start: float) -> None:
    import torch
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print("=== Loading data (KGE) ===")
    max_train = args.max_train if args.max_train is not None else args.max_samples
    max_valid = args.max_valid if args.max_valid is not None else args.max_samples
    max_test = args.max_test if args.max_test is not None else args.max_samples
    train_df, valid_df, test_df = load_splits(
        max_train=max_train,
        max_valid=max_valid,
        max_test=max_test,
    )

    train_entities = set(train_df["head"]).union(train_df["tail"])
    valid_entities = set(valid_df["head"]).union(valid_df["tail"])
    test_entities = set(test_df["head"]).union(test_df["tail"])
    all_entities = train_entities.union(valid_entities).union(test_entities)
    all_relations = set(train_df["relation"]).union(valid_df["relation"]).union(test_df["relation"])

    print("=== Dataset stats ===")
    print(
        f"Entities: {len(all_entities)} | Relations: {len(all_relations)} | "
        f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}"
    )
    print(f"Entities in valid not in train: {len(valid_entities - train_entities)}")
    print(f"Entities in test not in train: {len(test_entities - train_entities)}")

    if (valid_entities - train_entities) or (test_entities - train_entities):
        if args.no_filter_unseen:
            print("=== Warning: KGE requires entities seen in train; filtering unseen entities anyway ===")
        else:
            print("=== KGE requires entities seen in train; filtering unseen entities ===")
        before_valid = len(valid_df)
        before_test = len(test_df)
        valid_df = valid_df[
            valid_df["head"].isin(train_entities) & valid_df["tail"].isin(train_entities)
        ]
        test_df = test_df[
            test_df["head"].isin(train_entities) & test_df["tail"].isin(train_entities)
        ]
        print(f"=== Filtered {before_valid - len(valid_df)} valid triples with unseen entities ===")
        print(f"=== Filtered {before_test - len(test_df)} test triples with unseen entities ===")

    train_triples = train_df[["head", "relation", "tail"]].to_numpy(dtype=str)
    valid_triples = valid_df[["head", "relation", "tail"]].to_numpy(dtype=str)
    test_triples = test_df[["head", "relation", "tail"]].to_numpy(dtype=str)

    train_tf = TriplesFactory.from_labeled_triples(train_triples)
    valid_tf = TriplesFactory.from_labeled_triples(
        valid_triples,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        test_triples,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== Training ComplEx (KGE) ===")
    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model="ComplEx",
        model_kwargs={"embedding_dim": args.complex_dim},
        training_kwargs={"num_epochs": args.complex_epochs, "batch_size": args.complex_batchsize},
        optimizer_kwargs={"lr": args.complex_lr},
        training_loop="sLCWA",
        device=str(device),
        random_seed=42,
    )

    evaluator = RankBasedEvaluator(filtered=True)
    eval_device = device
    if args.complex_eval_device != "auto":
        eval_device = torch.device(args.complex_eval_device)
    if eval_device != device:
        result.model.to(eval_device)
    print("=== Validation ranking metrics (ComplEx) ===")
    val_results = evaluator.evaluate(
        model=result.model,
        mapped_triples=valid_tf.mapped_triples,
        additional_filter_triples=[train_tf.mapped_triples],
        batch_size=args.complex_eval_batchsize,
        device=eval_device,
        slice_size=args.complex_eval_slice_size,
    )
    val_metrics = _metric_results_to_dict(val_results)
    print(val_metrics)

    print("=== Test ranking metrics (ComplEx) ===")
    test_results = evaluator.evaluate(
        model=result.model,
        mapped_triples=test_tf.mapped_triples,
        additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
        batch_size=args.complex_eval_batchsize,
        device=eval_device,
        slice_size=args.complex_eval_slice_size,
    )
    test_metrics = _metric_results_to_dict(test_results)
    print(test_metrics)

    metrics = {
        "model": "complex-kge",
        "device": args.device,
        "max_train": max_train,
        "max_valid": max_valid,
        "max_test": max_test,
        "val_link_prediction": val_metrics,
        "test_link_prediction": test_metrics,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"=== Wrote metrics to {output_path} ===")
    print("=== Metrics summary ===")
    print(json.dumps(metrics, indent=2))


def run_experiment(args: argparse.Namespace) -> None:
    start = time.time()
    print("=== Starting experiment ===")
    print(f"Model: {args.model} | device={args.device}")
    print(
        "Splits cap -> train: "
        f"{args.max_train}, valid: {args.max_valid}, test: {args.max_test}"
    )

    try:
        if args.model == "complex-kge":
            _run_complex_kge(args, start)
            return

        print("=== Loading data (small experiment) ===")
        max_train = args.max_train if args.max_train is not None else args.max_samples
        max_valid = args.max_valid if args.max_valid is not None else args.max_samples
        max_test = args.max_test if args.max_test is not None else args.max_samples
        if args.max_samples is not None:
            print(
                "Using max-samples cap for splits -> "
                f"train: {max_train}, valid: {max_valid}, test: {max_test}"
            )
        X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(
            max_train=max_train,
            max_valid=max_valid,
            max_test=max_test,
            n_neg_per_pos=args.n_neg_per_pos,
            n_neg_per_pos_eval=args.n_neg_eval,
            filter_unseen=not args.no_filter_unseen,
            hard_negatives=args.hard_negatives,
            corrupt_head_prob=args.corrupt_head_prob,
        )
        if args.overfit_small:
            X_valid, y_valid = X_train.copy(), y_train.copy()
            X_test, y_test = X_train.copy(), y_train.copy()
            print("=== Overfit sanity mode on 200 triples (train == valid == test) ===")

        if args.model == "tabicl":
            print("=== Building TabICL ===")
            clf = build_tabicl()
        elif args.model == "tabpfn":
            print("=== Building TabPFN ===")
            clf = build_tabpfn(device=args.device)
        elif args.model == "limix":
            print("=== Building LimiX ===")
            clf = build_limix(
                device=args.device,
                model_path=args.limix_model_path,
                limix_path=args.limix_path,
                model_id=args.limix_model_id,
                model_file=args.limix_model_file,
                cache_dir=args.limix_cache_dir,
                inference_config=args.limix_config,
            )
        elif args.model == "tabdpt":
            print("=== Building TabDPT ===")
            clf = build_tabdpt(
                device=None if args.device == "auto" else args.device,
                model_weight_path=args.tabdpt_weights,
                tabdpt_path=args.tabdpt_path,
                n_ensembles=args.tabdpt_n_ensembles,
                temperature=args.tabdpt_temperature,
                context_size=args.tabdpt_context_size,
                permute_classes=args.tabdpt_permute_classes,
                seed=42,
            )
        elif args.model == "saint":
            print("=== Building SAINT ===")
            clf = build_saint(
                device=None if args.device == "auto" else args.device,
                saint_path=args.saint_path,
                embedding_size=args.saint_embedding_size,
                transformer_depth=args.saint_depth,
                attention_heads=args.saint_heads,
                attentiontype=args.saint_attentiontype,
                lr=args.saint_lr,
                epochs=args.saint_epochs,
                batchsize=args.saint_batchsize,
                seed=42,
            )
        elif args.model == "kgbert":
            print("=== Building KG-BERT ===")
            clf = build_kgbert(
                model_name=args.kgbert_model,
                device=None if args.device == "auto" else args.device,
                max_length=args.kgbert_max_length,
                lr=args.kgbert_lr,
                epochs=args.kgbert_epochs,
                batchsize=args.kgbert_batchsize,
                seed=42,
            )
        elif args.model == "rotatee":
            print("=== Building RotatE ===")
            clf = build_rotatee(
                embedding_dim=args.rotatee_dim,
                epochs=args.rotatee_epochs,
                batchsize=args.rotatee_batchsize,
                device=None if args.device == "auto" else args.device,
                lr=1e-3,
                seed=42,
            )
        elif args.model == "complex":
            print("=== Building ComplEx ===")
            clf = build_complex(
                embedding_dim=args.complex_dim,
                epochs=args.complex_epochs,
                batchsize=args.complex_batchsize,
                device=None if args.device == "auto" else args.device,
                lr=args.complex_lr,
                seed=42,
            )
        elif args.model == "tag":
            print("=== Building TAG ===")
            clf = build_tag(
                tag_path=args.tag_path,
                base_model=args.tag_base_model,
                device=None if args.device == "auto" else args.device,
                max_train_rows=args.tag_max_train_rows,
                max_cells_per_batch=args.tag_max_cells_per_batch,
                seed=42,
            )
        elif args.model == "tag-graph":
            print("=== Building TAG (graph features) ===")
            feature_list = [f.strip() for f in args.tag_graph_features.split(",") if f.strip()]
            clf = build_tag_graph(
                tag_path=args.tag_path,
                base_model=args.tag_base_model,
                device=None if args.device == "auto" else args.device,
                node_feature_dim=args.tag_graph_node_dim,
                relation_feature_dim=args.tag_graph_relation_dim,
                n_hops=args.tag_graph_hops,
                feature_names=feature_list,
                include_gpse=args.tag_graph_gpse,
                max_train_rows=args.tag_max_train_rows,
                max_cells_per_batch=args.tag_max_cells_per_batch,
                seed=42,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        print("=== Fitting ===")
        clf.fit(X_train, y_train)

        print("=== Task definition ===")
        print("Binary link prediction (head, relation, tail -> true/false). Classes: 2")

        if args.classification_metrics:
            print("=== Validation classification metrics ===")
            val_cls = binary_classification_metrics(
                clf, X_valid, y_valid, threshold=args.classification_threshold
            )
            print(val_cls)
            print("=== Test classification metrics ===")
            test_cls = binary_classification_metrics(
                clf, X_test, y_test, threshold=args.classification_threshold
            )
            print(test_cls)

        train_pos = X_train[y_train == 1].copy()
        valid_pos = X_valid[y_valid == 1].copy()
        test_pos = X_test[y_test == 1].copy()
        all_pos = pd.concat([train_pos, valid_pos, test_pos], ignore_index=True)

        candidate_entities = sorted(set(train_pos["head"]).union(train_pos["tail"]))
        print(f"=== Candidate entities (train): {len(candidate_entities)} ===")

        print("=== Validation ranking metrics (tail prediction) ===")
        val_lp_tail = filtered_ranking_metrics_binary(
            clf,
            valid_pos,
            candidate_entities,
            all_pos,
            predict="tail",
            batch_size=args.ranking_batchsize,
            max_candidates=args.max_candidates,
            seed=args.candidate_seed,
        )
        print(val_lp_tail)

        print("=== Validation ranking metrics (head prediction) ===")
        val_lp_head = filtered_ranking_metrics_binary(
            clf,
            valid_pos,
            candidate_entities,
            all_pos,
            predict="head",
            batch_size=args.ranking_batchsize,
            max_candidates=args.max_candidates,
            seed=args.candidate_seed,
        )
        print(val_lp_head)

        print("=== Test ranking metrics (tail prediction) ===")
        test_lp_tail = filtered_ranking_metrics_binary(
            clf,
            test_pos,
            candidate_entities,
            all_pos,
            predict="tail",
            batch_size=args.ranking_batchsize,
            max_candidates=args.max_candidates,
            seed=args.candidate_seed,
        )
        print(test_lp_tail)

        print("=== Test ranking metrics (head prediction) ===")
        test_lp_head = filtered_ranking_metrics_binary(
            clf,
            test_pos,
            candidate_entities,
            all_pos,
            predict="head",
            batch_size=args.ranking_batchsize,
            max_candidates=args.max_candidates,
            seed=args.candidate_seed,
        )
        print(test_lp_head)

        if args.sampled_ranking_metrics:
            sampled_neg = (
                args.n_neg_eval
                if args.sampled_ranking_negatives is None
                else args.sampled_ranking_negatives
            )
            print("=== Validation sampled ranking metrics (tail prediction) ===")
            val_lp_tail_sampled = sampled_ranking_metrics_binary(
                clf,
                valid_pos,
                candidate_entities,
                all_pos,
                n_neg_per_pos=sampled_neg,
                predict="tail",
            )
            print(val_lp_tail_sampled)

            print("=== Validation sampled ranking metrics (head prediction) ===")
            val_lp_head_sampled = sampled_ranking_metrics_binary(
                clf,
                valid_pos,
                candidate_entities,
                all_pos,
                n_neg_per_pos=sampled_neg,
                predict="head",
            )
            print(val_lp_head_sampled)

            print("=== Test sampled ranking metrics (tail prediction) ===")
            test_lp_tail_sampled = sampled_ranking_metrics_binary(
                clf,
                test_pos,
                candidate_entities,
                all_pos,
                n_neg_per_pos=sampled_neg,
                predict="tail",
            )
            print(test_lp_tail_sampled)

            print("=== Test sampled ranking metrics (head prediction) ===")
            test_lp_head_sampled = sampled_ranking_metrics_binary(
                clf,
                test_pos,
                candidate_entities,
                all_pos,
                n_neg_per_pos=sampled_neg,
                predict="head",
            )
            print(test_lp_head_sampled)

        metrics = {
            "model": args.model,
            "device": args.device,
            "max_train": max_train,
            "max_valid": max_valid,
            "max_test": max_test,
            "overfit_small": args.overfit_small,
            "val_link_prediction_tail": val_lp_tail,
            "val_link_prediction_head": val_lp_head,
            "test_link_prediction_tail": test_lp_tail,
            "test_link_prediction_head": test_lp_head,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        if args.classification_metrics:
            metrics["val_classification"] = val_cls
            metrics["test_classification"] = test_cls
        if args.sampled_ranking_metrics:
            metrics["val_sampled_link_prediction_tail"] = val_lp_tail_sampled
            metrics["val_sampled_link_prediction_head"] = val_lp_head_sampled
            metrics["test_sampled_link_prediction_tail"] = test_lp_tail_sampled
            metrics["test_sampled_link_prediction_head"] = test_lp_head_sampled
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"=== Wrote metrics to {output_path} ===")
        print("=== Metrics summary ===")
        print(json.dumps(metrics, indent=2))
    except Exception:
        print("=== Experiment failed ===")
        traceback.print_exc()
        raise
