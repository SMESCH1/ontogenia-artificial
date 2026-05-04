"""CLI: smoke test y evaluaciones reproducibles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ontogenia.checkpoints import (
    CHECKPOINT_STEPS,
    PYTHIA_MODEL_IDS,
    model_args_for_pythia,
    revision_for_step,
)
from ontogenia.harness import build_result_envelope, run_lm_eval, save_json


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
            log_samples=args.log_samples,
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
        save_json(out_path, env)
        print(json.dumps({"saved": str(out_path.resolve())}, indent=2))

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
                log_samples=args.log_samples,
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
            save_json(out_path, env)
            msg = {"saved": str(out_path.resolve()), "size": size, "step": step}
            print(json.dumps(msg, indent=2))

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Una corrida genérica (para scripts de sweep)."""
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    device = args.device or _default_device()
    raw = run_lm_eval(
        model_args=args.model_args,
        tasks=tasks,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
        log_samples=args.log_samples,
        bootstrap_iters=args.bootstrap_iters,
    )
    meta = json.loads(args.meta_json) if args.meta_json else {}
    out = {
        "meta": meta,
        "lm_eval": raw,
    }
    save_json(Path(args.output_path), out)
    print(json.dumps({"saved": args.output_path}, indent=2))
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
    ps.add_argument("--log-samples", action="store_true")
    ps.add_argument("--output-dir", default="results", help="Raíz de resultados (gitignored)")

    pe = sub.add_parser("eval", help="Evaluación genérica (model_args explícito)")
    pe.add_argument("--model-args", required=True, help="Cadena lm-eval hf model_args")
    pe.add_argument("--tasks", required=True, help="Lista separada por comas")
    pe.add_argument("--output-path", required=True)
    pe.add_argument("--limit", type=float, default=None)
    pe.add_argument("--batch-size", default="auto")
    pe.add_argument("--device", default=None)
    pe.add_argument("--log-samples", action="store_true")
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
    pw.add_argument("--bootstrap-iters", type=int, default=0)
    pw.add_argument("--output-dir", default="results")

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
    raise SystemExit(2)


if __name__ == "__main__":
    main()
