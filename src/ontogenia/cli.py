from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from ontogenia.aggregate import aggregate_results_dir, aggregate_samples_dir
from ontogenia.checkpoints import (
    CHECKPOINT_STEPS,
    PYTHIA_MODEL_IDS,
    model_args_for_pythia,
    revision_for_step,
)
from ontogenia.harness import (
    build_result_envelope,
    detach_lm_eval_samples,
    run_lm_eval,
    save_json,
)
from ontogenia.human_alignment import run_human_alignment
from ontogenia.prefetch import prefetch_checkpoints
from ontogenia.topology import classify_all_tasks

# CLI: smoke test y evaluaciones reproducibles.
# https://arxiv.org/abs/2304.01373
# modelo en Hub (ej. pythia-70m / familia), buscar la tarjeta del tamaño que uses
# (pythia-14m-deduped, etc.) para la config exacta.


def _persist_eval_result(out_path: Path, env: dict, write_samples_to: Path | None) -> None:
    if write_samples_to is not None:
        detach_lm_eval_samples(env, write_samples_to)
    save_json(out_path, env)


def _default_device() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return None


def cmd_smoke(args: argparse.Namespace) -> int:
    """Pocos checkpoints × una tarea BLiMP — validación del pipeline."""
    size = args.model_size
    pretrained = PYTHIA_MODEL_IDS[size]
    steps = [int(x) for x in args.steps.split(",")]
    tasks = [args.task]

    out_root = Path(args.output_dir)
    device = args.device or _default_device()
    log_samples = bool(args.log_samples or args.samples_dir)

    for step in steps:
        rev = revision_for_step(step)
        margs = model_args_for_pythia(size, step, dtype=args.dtype)
        print(f"[smoke] {pretrained} @ {rev} tasks={tasks} device={device}", file=sys.stderr)
        raw = run_lm_eval(
            model_args=margs,
            tasks=tasks,
            batch_size=args.batch_size,
            device=device,
            limit=args.limit,
            log_samples=log_samples,
            bootstrap_iters=0,
        )
        env = build_result_envelope(
            raw,
            model_size=size,
            pretrained=pretrained,
            revision=rev,
            tasks=tasks,
        )
        fname = f"{size}_step{step}_{args.task}.json"
        out_path = out_root / "smoke" / fname
        sidecar = None
        if args.samples_dir:
            sidecar = Path(args.samples_dir).expanduser() / f"{out_path.stem}_samples.json"
        _persist_eval_result(out_path, env, sidecar)
        out_msg = {"saved": str(out_path.resolve())}
        if sidecar and sidecar.exists():
            out_msg["samples"] = str(sidecar.resolve())
        print(json.dumps(out_msg, indent=2))

    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """
    Semana 3: una corrida lm-eval por (modelo, checkpoint).

    No recarga el modelo entre tareas del mismo grupo (ej. `blimp` incluye subtasks).
    """
    sizes = [x.strip() for x in args.sizes.split(",") if x.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.steps == "default":
        steps = list(CHECKPOINT_STEPS)
    else:
        steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]

    out_root = Path(args.output_dir)
    device = args.device or _default_device()
    log_samples = bool(args.log_samples or args.samples_dir)

    for size in sizes:
        if size not in PYTHIA_MODEL_IDS:
            print(f"error: tamaño desconocido {size}", file=sys.stderr)
            return 2
        pretrained = PYTHIA_MODEL_IDS[size]
        for step in steps:
            rev = revision_for_step(step)
            margs = model_args_for_pythia(size, step, dtype=args.dtype)
            print(f"[sweep] {pretrained} @ {rev} tasks={tasks} device={device}", file=sys.stderr)
            raw = run_lm_eval(
                model_args=margs,
                tasks=tasks,
                batch_size=args.batch_size,
                device=device,
                limit=args.limit,
                log_samples=log_samples,
                bootstrap_iters=args.bootstrap_iters,
            )
            env = build_result_envelope(
                raw,
                model_size=size,
                pretrained=pretrained,
                revision=rev,
                tasks=tasks,
            )
            fname = f"{size}_{rev}.json"
            out_path = out_root / "sweep" / fname
            sidecar = None
            if args.samples_dir:
                sidecar = Path(args.samples_dir).expanduser() / f"{out_path.stem}_samples.json"
            _persist_eval_result(out_path, env, sidecar)
            msg = {"saved": str(out_path.resolve()), "size": size, "step": step}
            if sidecar and sidecar.exists():
                msg["samples"] = str(sidecar.resolve())
            print(json.dumps(msg, indent=2))

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Una corrida genérica (para scripts de sweep)."""
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    device = args.device or _default_device()
    log_samples = bool(args.log_samples or args.samples_dir)
    raw = run_lm_eval(
        model_args=args.model_args,
        tasks=tasks,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
        log_samples=log_samples,
        bootstrap_iters=args.bootstrap_iters,
    )
    meta = json.loads(args.meta_json) if args.meta_json else {}
    out = {
        "meta": meta,
        "lm_eval": raw,
    }
    out_path = Path(args.output_path)
    sidecar = None
    if args.samples_dir:
        sidecar = Path(args.samples_dir).expanduser() / f"{out_path.stem}_samples.json"
    _persist_eval_result(out_path, out, sidecar)
    msg = {"saved": str(out_path.resolve())}
    if sidecar and sidecar.exists():
        msg["samples"] = str(sidecar.resolve())
    print(json.dumps(msg, indent=2))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    if args.sweep_only and args.smoke_only:
        print("error: no usar --sweep-only y --smoke-only juntos", file=sys.stderr)
        return 2
    root = Path(args.results_dir)
    out = Path(args.output_parquet)
    aggregate_results_dir(
        root,
        out_parquet=out,
        include_smoke=args.smoke_only or not args.sweep_only,
        include_sweep=args.sweep_only or not args.smoke_only,
    )
    print(json.dumps({"saved": str(out.resolve()), "rows_hint": "see parquet"}, indent=2))
    return 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    sizes = [x.strip() for x in args.sizes.split(",") if x.strip()]
    if args.steps == "default":
        steps = list(CHECKPOINT_STEPS)
    else:
        steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    results = prefetch_checkpoints(sizes=sizes, steps=steps, cache_dir=args.cache_dir)
    failed = [r for r in results if not r.ok]
    msg = {
        "total": len(results),
        "ok": len(results) - len(failed),
        "failed": len(failed),
    }
    if failed:
        msg["first_error"] = {
            "model_size": failed[0].model_size,
            "revision": failed[0].revision,
            "error": failed[0].error,
        }
    print(json.dumps(msg, indent=2))
    return 1 if failed else 0


def cmd_topology(args: argparse.Namespace) -> int:
    frame = pd.read_parquet(args.metrics_parquet)
    summary = classify_all_tasks(frame, metric_col=args.metric_col)
    out = Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out, index=False)
    print(
        json.dumps(
            {
                "saved": str(out.resolve()),
                "rows": int(len(summary)),
                "shapes": summary["shape"].value_counts(dropna=False).to_dict(),
            },
            indent=2,
        )
    )
    return 0


def cmd_human_alignment(args: argparse.Namespace) -> int:
    result = run_human_alignment(
        metrics_parquet=args.metrics_parquet,
        aoa_csv=args.aoa_csv,
        metric_col=args.metric_col,
        target_model_size=args.model_size,
        aoa_task_col=args.aoa_task_col,
        aoa_months_col=args.aoa_months_col,
        n_boot=args.bootstrap_iters,
    )
    stats_out = Path(args.output_stats_json)
    overlap_out = Path(args.output_overlap_parquet)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    overlap_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(result.stats.to_json(orient="records", indent=2), encoding="utf-8")
    result.overlap.to_parquet(overlap_out, index=False)
    print(
        json.dumps(
            {
                "stats": str(stats_out.resolve()),
                "overlap": str(overlap_out.resolve()),
                "n_overlap": int(len(result.overlap)),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ontogenia", description="Pipeline Ontogenia / lm-eval")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("smoke", help="Smoke test Pythia × pocos checkpoints × 1 tarea BLiMP")
    ps.add_argument("--model-size", default="14m", choices=sorted(PYTHIA_MODEL_IDS.keys()))
    ps.add_argument(
        "--steps",
        default="0,512,2000",
        help="Pasos de entrenamiento separados por coma",
    )
    ps.add_argument(
        "--task",
        default="blimp_anaphor_number_agreement",
        help="Nombre de tarea lm-eval (ej. blimp_anaphor_number_agreement)",
    )
    ps.add_argument("--limit", type=int, default=20, help="Items por tarea (acortar en pruebas)")
    ps.add_argument("--batch-size", default="auto")
    ps.add_argument("--dtype", default="float16")
    ps.add_argument("--device", default=None)
    ps.add_argument(
        "--log-samples",
        action="store_true",
        help=(
            "Incluye por-ítem en lm-eval (doc, resps, sentence_good/bad en BLiMP); "
            "JSON mucho más grande"
        ),
    )
    ps.add_argument(
        "--samples-dir",
        default=None,
        metavar="DIR",
        help=(
            "Activa log-samples; guarda lm_eval['samples'] en "
            "DIR/<stem>_samples.json y los quita del JSON principal"
        ),
    )
    ps.add_argument("--output-dir", default="results", help="Raíz de resultados (gitignored)")

    pe = sub.add_parser("eval", help="Evaluación genérica (model_args explícito)")
    pe.add_argument("--model-args", required=True, help="Cadena lm-eval hf model_args")
    pe.add_argument("--tasks", required=True, help="Lista separada por comas")
    pe.add_argument("--output-path", required=True)
    pe.add_argument("--limit", type=float, default=None)
    pe.add_argument("--batch-size", default="auto")
    pe.add_argument("--device", default=None)
    pe.add_argument("--log-samples", action="store_true")
    pe.add_argument(
        "--samples-dir",
        default=None,
        metavar="DIR",
        help="Igual que en smoke: sidecar con muestras por ítem",
    )
    pe.add_argument("--bootstrap-iters", type=int, default=0)
    pe.add_argument("--meta-json", default=None, help='JSON extra para envelope meta, ej. "{}"')

    pw = sub.add_parser(
        "sweep",
        help="Barrido completo (modelo × checkpoints × grupo de tareas; GPU intensivo)",
    )
    pw.add_argument(
        "--sizes",
        default="14m,160m,410m",
        help="Tamaños separados por coma",
    )
    pw.add_argument(
        "--steps",
        default="default",
        help='"default" = 24 pasos del ROADMAP; o lista explicita e.g. 0,512,2000',
    )
    pw.add_argument(
        "--tasks",
        default="blimp",
        help="Grupo o lista lm-eval, ej. blimp o blimp_anaphor_number_agreement",
    )
    pw.add_argument("--limit", type=float, default=None)
    pw.add_argument("--batch-size", default="auto")
    pw.add_argument("--dtype", default="float16")
    pw.add_argument("--device", default=None)
    pw.add_argument("--log-samples", action="store_true")
    pw.add_argument(
        "--samples-dir",
        default=None,
        metavar="DIR",
        help="Sidecar *_samples.json por corrida (recomendado en sweep grande)",
    )
    pw.add_argument("--bootstrap-iters", type=int, default=0)
    pw.add_argument("--output-dir", default="results")

    pa = sub.add_parser(
        "aggregate",
        help="Consolidar métricas agregadas de JSON en results/ → Parquet",
    )
    pa.add_argument("--results-dir", default="results", help="Raíz que contiene sweep/ y smoke/")
    pa.add_argument(
        "--output-parquet",
        default="results/aggregated_metrics.parquet",
        help="Ruta del Parquet de salida",
    )
    pa.add_argument(
        "--sweep-only",
        action="store_true",
        help="Sólo leer results/sweep/",
    )
    pa.add_argument(
        "--smoke-only",
        action="store_true",
        help="Sólo leer results/smoke/",
    )
    pa.add_argument(
        "--samples-dir",
        default=None,
        metavar="DIR",
        help="Si se indica, agrega `*_samples.json` y escribe `--output-samples-parquet`",
    )
    pa.add_argument(
        "--output-samples-parquet",
        default="results/aggregated_samples.parquet",
        help="Parquet por-ítem con margen SLLN (requiere --samples-dir)",
    )
    pa.add_argument(
        "--slln-alpha",
        type=float,
        default=0.5,
        help="Alpha para margen SLLN en agregación por-ítem",
    )

    pp = sub.add_parser(
        "prefetch",
        help="Pre-descarga checkpoints Pythia al cache local de HuggingFace",
    )
    pp.add_argument("--sizes", default="14m,160m,410m")
    pp.add_argument(
        "--steps",
        default="default",
        help='"default" = 24 pasos; o lista explícita, ej. 0,512,2000',
    )
    pp.add_argument("--cache-dir", default=None)

    pt = sub.add_parser(
        "topology",
        help="Clasifica topología por tarea desde aggregated_metrics.parquet",
    )
    pt.add_argument(
        "--metrics-parquet",
        default="results/aggregated_metrics.parquet",
    )
    pt.add_argument("--metric-col", default="acc,none")
    pt.add_argument(
        "--output-parquet",
        default="results/topology_summary.parquet",
    )

    ph = sub.add_parser(
        "human-alignment",
        help="Calcula Spearman step_estabilización vs AoA (Wordbank)",
    )
    ph.add_argument(
        "--metrics-parquet",
        default="results/aggregated_metrics.parquet",
    )
    ph.add_argument("--aoa-csv", default="data/human_milestones.csv")
    ph.add_argument("--metric-col", default="acc,none")
    ph.add_argument("--model-size", default="160m")
    ph.add_argument("--aoa-task-col", default="task")
    ph.add_argument("--aoa-months-col", default="aoa_months")
    ph.add_argument("--bootstrap-iters", type=int, default=10000)
    ph.add_argument(
        "--output-stats-json",
        default="results/human_alignment_stats.json",
    )
    ph.add_argument(
        "--output-overlap-parquet",
        default="results/human_alignment_overlap.parquet",
    )

    return p


def main() -> None:
    # Evita fragmentación GPU en algunos entornos
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "smoke":
        raise SystemExit(cmd_smoke(args))
    if args.command == "eval":
        raise SystemExit(cmd_eval(args))
    if args.command == "sweep":
        raise SystemExit(cmd_sweep(args))
    if args.command == "aggregate":
        code = cmd_aggregate(args)
        if code == 0 and args.samples_dir:
            aggregate_samples_dir(
                results_root=Path(args.results_dir),
                samples_dir=Path(args.samples_dir),
                out_parquet=Path(args.output_samples_parquet),
                include_smoke=args.smoke_only or not args.sweep_only,
                include_sweep=args.sweep_only or not args.smoke_only,
                alpha=args.slln_alpha,
            )
            print(
                json.dumps(
                    {"saved_samples": str(Path(args.output_samples_parquet).resolve())},
                    indent=2,
                )
            )
        raise SystemExit(code)
    if args.command == "prefetch":
        raise SystemExit(cmd_prefetch(args))
    if args.command == "topology":
        raise SystemExit(cmd_topology(args))
    if args.command == "human-alignment":
        raise SystemExit(cmd_human_alignment(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
