#!/usr/bin/env python
"""H-clock / L-clock deterministic dynamics for five selected core conditions.

The H-clock is the primary transport clock: every HRM block contains ``eval_l``
L updates followed by one H update.  All 160 H updates are analysed (there is no
burn-in); four segments contain 40 H updates each.  L is additionally analysed
on its native lower-update clock and through block-local execution summaries.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import filecmp
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scripts.analyze_long_rollout_msd import RunDirectory, build_model, data_kwargs, load_module
from scripts.core_five_l_depth_long_rollout import (
    absolute_checkpoint, advance_hrm_l, advance_rt_l, atomic_npz, initial_hrm_state,
    initial_rt_state,
)
from scripts.core_five_long_rollout import core_runs, fixed_random_samples, read_csv
from scripts.long_rollout_msd_utils import rt_state_msd_per_puzzle, segment_for_pair, state_msd_per_puzzle


TOTAL_H_UPDATES = 160
H_BOUNDARIES = np.asarray((0, 40, 80, 120, 160), dtype=np.int64)
H_LAGS = np.asarray((1, 2, 3, 4, 6, 8, 12, 16), dtype=np.int64)
L_LAGS = np.asarray((1, 2, 4, 8, 16, 32, 64), dtype=np.int64)
PHASE_BINS = 16
SCHEME = "core_h_l_clock_all_160_h_updates_v1"
CORE_KEYS = (("H2L6_h", 6), ("H2L6_h", 32), ("H2L6_h", 512), ("H2L1_h", 1), ("RT", 1))

SELECTION_FIELDS = ["kind", "condition", "readout", "train_l", "seed", "epoch", "checkpoint", "test_exact_match", "cell_accuracy", "evaluated_examples", "selection_completed"]
METADATA_FIELDS = ["kind", "condition", "eval_l", "seed", "checkpoint", "best_epoch", "samples", "sample_seed", "sample_manifest_sha256", "total_h_updates", "total_l_updates", "h_boundaries", "h_lags", "l_lags", "per_puzzle_file"]
MSD_FIELDS = ["condition", "eval_l", "seed", "clock", "state", "segment", "lag", "mean", "median", "ci95_low", "ci95_high", "puzzles", "origins"]
EXEC_FIELDS = ["condition", "eval_l", "seed", "segment", "metric", "mean", "median", "ci95_low", "ci95_high", "puzzles"]
PHASE_FIELDS = ["condition", "eval_l", "seed", "segment", "phase_bin", "phase_center", "mean", "median", "ci95_low", "ci95_high", "puzzles"]
SEED_MSD_FIELDS = ["condition", "eval_l", "clock", "state", "segment", "lag", "seed_mean", "seed_sd", "seeds"]
SEED_EXEC_FIELDS = ["condition", "eval_l", "segment", "metric", "seed_mean", "seed_sd", "seeds"]


@dataclass(frozen=True)
class Unit:
    run: RunDirectory
    eval_l: int
    @property
    def key(self) -> tuple[str, int, int]: return self.run.condition, self.run.seed, self.eval_l
    @property
    def label(self) -> str: return f"{self.run.condition}/L{self.eval_l}/seed_{self.run.seed}"


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def units(runs: list[RunDirectory], seeds: tuple[int, ...]) -> list[Unit]:
    lookup = {(run.condition, run.seed): run for run in runs}
    missing = [f"{name}/seed_{seed}" for name, _ in CORE_KEYS for seed in seeds if (name, seed) not in lookup]
    if missing: raise FileNotFoundError("Missing core H/L-clock runs:\n" + "\n".join(missing))
    return [Unit(lookup[(name, seed)], l_value) for name, l_value in CORE_KEYS for seed in seeds]


def selected(path: Path, selected_units: list[Unit]) -> dict[tuple[str, int], dict[str, str]]:
    rows = {(row["condition"], int(row["seed"])): row for row in read_csv(path)}
    needed = {(unit.run.condition, unit.run.seed) for unit in selected_units}
    if not needed <= rows.keys(): raise FileNotFoundError(f"Missing selected checkpoints in {path}: {sorted(needed - rows.keys())}")
    for key in needed:
        if not absolute_checkpoint(rows[key]["checkpoint"]).is_file(): raise FileNotFoundError(rows[key]["checkpoint"])
    return {key: rows[key] for key in needed}


def write_manifest(output: Path, indices: np.ndarray, sample_seed: int) -> str:
    rows = [{"sample_position": i, "test_hard_stream_index": int(v), "sample_seed": sample_seed} for i, v in enumerate(indices)]
    atomic_csv(output / "sample_manifest.csv", rows, ["sample_position", "test_hard_stream_index", "sample_seed"])
    digest = hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()
    (output / "sample_manifest.sha256").write_text(digest + "\n")
    return digest


@torch.inference_mode()
def advance_block(model: torch.nn.Module, state: tuple[torch.Tensor, torch.Tensor], embedding: torch.Tensor, eval_l: int) -> tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
    """Advance one H block and return all L-step displacements in the block."""
    h, l = state; l_deltas: list[torch.Tensor] = []
    for step in range(eval_l):
        old_l = l
        l = model.L_level(l + h + embedding)  # type: ignore[attr-defined]
        l_deltas.append(l - old_l)
    h = model.H_level(h + l)  # type: ignore[attr-defined]
    return (h, l), l_deltas


def _pair_hrm(model: torch.nn.Module, x: torch.Tensor, eval_l: int, lags: np.ndarray, boundaries: np.ndarray, *, clock: str, progress: tqdm[Any]) -> tuple[np.ndarray, np.ndarray]:
    embedding = model.embed(x)  # type: ignore[attr-defined]
    total = int(boundaries[-1])
    values = np.full((x.shape[0], 2, 4, len(lags)), np.nan, np.float32); origins = np.zeros((4, len(lags)), np.int32)
    for p, lag in enumerate(lags.tolist()):
        ref, lead = initial_hrm_state(model, x), initial_hrm_state(model, x)
        if clock == "h":
            for _ in range(lag): lead, _ = advance_block(model, lead, embedding, eval_l)
        else:
            lead_phase = 0
            for _ in range(lag): lead, lead_phase, _ = advance_hrm_l(model, lead, embedding, lead_phase, eval_l)
        totals = {s: [torch.zeros(x.shape[0], device=x.device), torch.zeros(x.shape[0], device=x.device)] for s in range(4)}; counts = np.zeros(4, np.int32)
        ref_phase = 0
        for time in range(total - lag + 1):
            segment = segment_for_pair(time, lag, tuple(boundaries.tolist()))
            if segment is not None:
                msd = state_msd_per_puzzle(*ref, *lead); totals[segment][0] += msd["h"]; totals[segment][1] += msd["l"]; counts[segment] += 1
            if time < total - lag:
                if clock == "h": ref, _ = advance_block(model, ref, embedding, eval_l); lead, _ = advance_block(model, lead, embedding, eval_l)
                else: ref, ref_phase, _ = advance_hrm_l(model, ref, embedding, ref_phase, eval_l); lead, lead_phase, _ = advance_hrm_l(model, lead, embedding, lead_phase, eval_l)
        for s in range(4):
            if counts[s]:
                origins[s, p] = counts[s]
                values[:, :, s, p] = torch.stack(totals[s]).transpose(0, 1).div(counts[s]).cpu().numpy()
        progress.update(1)
    return values, origins


def collect_execution(model: torch.nn.Module, x: torch.Tensor, eval_l: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-puzzle block execution statistics, all 160 H blocks included."""
    with torch.inference_mode():
        embedding = model.embed(x)  # type: ignore[attr-defined]
        state = initial_hrm_state(model, x); batch = x.shape[0]
        # block metrics: H jump, H jump cosine, L path, early/late step, H->next-L alignment, final L displacement
        values = torch.zeros(batch, 4, 7, device=x.device); counts = torch.zeros(4, 7, device=x.device)
        phase = torch.zeros(batch, 4, PHASE_BINS, device=x.device); phase_counts = torch.zeros(4, PHASE_BINS, device=x.device)
        previous_h_delta: torch.Tensor | None = None
        for block in range(TOTAL_H_UPDATES):
            segment = min(3, block // 40); h0, l0 = state
            state, l_deltas = advance_block(model, state, embedding, eval_l); h1, l1 = state
            h_delta = h1 - h0
            h_norm = torch.mean(h_delta.float().square(), dim=(-2, -1))
            values[:, segment, 0] += h_norm; counts[segment, 0] += 1
            if previous_h_delta is not None:
                cosine = torch.nn.functional.cosine_similarity(h_delta.flatten(1).float(), previous_h_delta.flatten(1).float(), dim=1)
                values[:, segment, 1] += cosine; counts[segment, 1] += 1
                first = l_deltas[0]
                align = torch.nn.functional.cosine_similarity(first.flatten(1).float(), previous_h_delta.flatten(1).float(), dim=1)
                values[:, segment, 5] += align; counts[segment, 5] += 1
            previous_h_delta = h_delta
            step_values = [torch.mean(delta.float().square(), dim=(-2, -1)) for delta in l_deltas]
            path = torch.stack(step_values).sum(0); final_disp = torch.mean((l1 - l0).float().square(), dim=(-2, -1))
            quarter = max(1, math.ceil(eval_l / 4))
            values[:, segment, 2] += path; values[:, segment, 3] += torch.stack(step_values[:quarter]).mean(0); values[:, segment, 4] += torch.stack(step_values[-quarter:]).mean(0); values[:, segment, 6] += final_disp
            counts[segment, [2, 3, 4, 6]] += 1
            for r, step_value in enumerate(step_values):
                b = min(PHASE_BINS - 1, (r * PHASE_BINS) // eval_l); phase[:, segment, b] += step_value; phase_counts[segment, b] += 1
        phase_mean = phase / phase_counts.clamp_min(1).unsqueeze(0)
        phase_mean[:, phase_counts == 0] = torch.nan
        return (values / counts.clamp_min(1).unsqueeze(0)).cpu().numpy(), phase_mean.cpu().numpy()


def collect_unit(unit: Unit, checkpoint_row: dict[str, str], fixed_x: torch.Tensor, metadata: dict[str, Any], args: Any, manifest: str, progress: tqdm[Any]) -> dict[str, object]:
    destination = args.output_dir / "per_puzzle" / f"{unit.run.condition}_evalL{unit.eval_l}_seed_{unit.run.seed}.npz"
    total_l = TOTAL_H_UPDATES * unit.eval_l
    h_bounds = H_BOUNDARIES
    l_bounds = H_BOUNDARIES * unit.eval_l
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as old:
            if str(old["scheme"].item()) == SCHEME: return metadata_row(unit, checkpoint_row, destination, len(fixed_x), args.sample_seed, manifest, total_l)
    model = build_model(unit.run, absolute_checkpoint(checkpoint_row["checkpoint"]), metadata, args.device)
    h_batches=[]; l_batches=[]; e_batches=[]; p_batches=[]; h_orig=None; l_orig=None
    chunks = math.ceil(len(fixed_x) / args.rollout_batch_size)
    for start in range(0, len(fixed_x), args.rollout_batch_size):
        x = fixed_x[start:start + args.rollout_batch_size].to(args.device, non_blocking=True)
        if unit.run.kind == "hrm":
            h, ho = _pair_hrm(model, x, unit.eval_l, H_LAGS, h_bounds, clock="h", progress=progress)
            usable_lags = L_LAGS[L_LAGS < 40 * unit.eval_l]
            l, lo = _pair_hrm(model, x, unit.eval_l, usable_lags, l_bounds, clock="l", progress=progress)
            e, p = collect_execution(model, x, unit.eval_l)
        else:
            h, ho, l, lo, e, p = collect_rt(model, x, progress)
        h_batches.append(h); l_batches.append(l); e_batches.append(e); p_batches.append(p)
        h_orig = ho if h_orig is None else h_orig; l_orig = lo if l_orig is None else l_orig
    assert h_orig is not None and l_orig is not None
    atomic_npz(destination, scheme=np.asarray(SCHEME), h_clock=np.concatenate(h_batches), h_origins=h_orig, h_lags=H_LAGS, l_clock=np.concatenate(l_batches), l_origins=l_orig, l_lags=usable_lags if unit.run.kind == "hrm" else L_LAGS, execution=np.concatenate(e_batches), phase=np.concatenate(p_batches), state_names=np.asarray(("h", "l") if unit.run.kind == "hrm" else ("rt",)))
    del model; torch.cuda.empty_cache() if args.device.type == "cuda" else None
    return metadata_row(unit, checkpoint_row, destination, len(fixed_x), args.sample_seed, manifest, total_l)


def collect_rt(model: torch.nn.Module, x: torch.Tensor, progress: tqdm[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        embedding=model.embed(x)  # type: ignore[attr-defined]
        def pair(lags: np.ndarray):
            out=np.full((x.shape[0],1,4,len(lags)),np.nan,np.float32); origins=np.zeros((4,len(lags)),np.int32)
            for p,lag in enumerate(lags):
                a=initial_rt_state(model,x); b=initial_rt_state(model,x)
                for _ in range(int(lag)): b=advance_rt_l(model,b,embedding)
                total={s:torch.zeros(x.shape[0],device=x.device) for s in range(4)}; count=np.zeros(4,np.int32)
                for t in range(TOTAL_H_UPDATES-int(lag)+1):
                    s=segment_for_pair(t,int(lag),tuple(H_BOUNDARIES.tolist()))
                    if s is not None: total[s]+=rt_state_msd_per_puzzle(a,b); count[s]+=1
                    if t<TOTAL_H_UPDATES-int(lag): a=advance_rt_l(model,a,embedding); b=advance_rt_l(model,b,embedding)
                for s in range(4):
                    if count[s]: out[:,0,s,p]=(total[s]/count[s]).cpu().numpy(); origins[s,p]=count[s]
                progress.update(1)
            return out,origins
        h,ho=pair(H_LAGS); l,lo=pair(L_LAGS)
        empty=np.full((x.shape[0],4,7),np.nan,np.float32); phase=np.full((x.shape[0],4,PHASE_BINS),np.nan,np.float32)
        return h,ho,l,lo,empty,phase


def metadata_row(unit: Unit, selected_row: dict[str, str], path: Path, samples: int, sample_seed: int, manifest: str, total_l: int) -> dict[str, object]:
    return {"kind":unit.run.kind,"condition":unit.run.condition,"eval_l":unit.eval_l,"seed":unit.run.seed,"checkpoint":selected_row["checkpoint"],"best_epoch":selected_row["epoch"],"samples":samples,"sample_seed":sample_seed,"sample_manifest_sha256":manifest,"total_h_updates":TOTAL_H_UPDATES,"total_l_updates":total_l,"h_boundaries":json.dumps(H_BOUNDARIES.tolist()),"h_lags":json.dumps(H_LAGS.tolist()),"l_lags":json.dumps(L_LAGS.tolist()),"per_puzzle_file":str(path)}


def bootstrap(values: np.ndarray, rng: np.random.Generator, repeats: int) -> tuple[float,float,float,float]:
    draws=values[rng.integers(0,len(values),size=(repeats,len(values)))].mean(1)
    return float(values.mean()),float(np.median(values)),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def summarize(output: Path, repeats: int, seed: int) -> tuple[list[dict[str,object]],list[dict[str,object]],list[dict[str,object]]]:
    rng=np.random.default_rng(seed); msd=[]; execution=[]; phase_rows=[]
    names=("h_jump_norm","h_jump_cosine","l_block_path","l_first_quarter_step","l_last_quarter_step","h_to_l_first_alignment","l_final_displacement")
    for meta in read_csv(output/"rollout_metadata.csv"):
        with np.load(meta["per_puzzle_file"],allow_pickle=False) as data:
            states=[str(x) for x in data["state_names"]]
            for clock,array_name,lag_name,origin_name in (("h","h_clock","h_lags","h_origins"),("l","l_clock","l_lags","l_origins")):
                arr=data[array_name]; lags=data[lag_name]; origins=data[origin_name]
                for si,state in enumerate(states):
                    for seg in range(4):
                        for p,lag in enumerate(lags):
                            v=arr[:,si,seg,p]
                            if not np.isfinite(v).all(): continue
                            mean,median,low,high=bootstrap(v,rng,repeats)
                            msd.append({"condition":meta["condition"],"eval_l":meta["eval_l"],"seed":meta["seed"],"clock":clock,"state":state,"segment":seg+1,"lag":int(lag),"mean":mean,"median":median,"ci95_low":low,"ci95_high":high,"puzzles":len(v),"origins":int(origins[seg,p])})
            for seg in range(4):
                for mi,name in enumerate(names):
                    v=data["execution"][:,seg,mi]
                    if not np.isfinite(v).all(): continue
                    mean,median,low,high=bootstrap(v,rng,repeats)
                    execution.append({"condition":meta["condition"],"eval_l":meta["eval_l"],"seed":meta["seed"],"segment":seg+1,"metric":name,"mean":mean,"median":median,"ci95_low":low,"ci95_high":high,"puzzles":len(v)})
                for bin_index in range(PHASE_BINS):
                    v=data["phase"][:,seg,bin_index]
                    if not np.isfinite(v).all(): continue
                    mean,median,low,high=bootstrap(v,rng,repeats)
                    phase_rows.append({"condition":meta["condition"],"eval_l":meta["eval_l"],"seed":meta["seed"],"segment":seg+1,"phase_bin":bin_index,"phase_center":(bin_index+.5)/PHASE_BINS,"mean":mean,"median":median,"ci95_low":low,"ci95_high":high,"puzzles":len(v)})
    return msd,execution,phase_rows


def make_figures(output: Path, msd: list[dict[str,object]], execution: list[dict[str,object]], phase_rows: list[dict[str,object]]) -> None:
    import pandas as pd
    d=pd.DataFrame(msd); e=pd.DataFrame(execution); colors={6:"tab:purple",32:"tab:blue",512:"tab:orange",1:"black"}
    for clock,state,title in (("h","h","H-clock H MSD"),("h","l","H-clock L MSD"),("l","l","L-clock L MSD")):
        figure,axes=plt.subplots(1,4,figsize=(15,3.4),sharey=True)
        for seg,axis in enumerate(axes,1):
            q=d[(d.clock==clock)&(d.segment==seg)&((d.state==state)|((state=="h")&(d.state=="rt")))]
            for (condition,l),g in q.groupby(["condition","eval_l"]):
                mean=g.groupby("lag")["mean"].mean().sort_index(); label="RT" if condition=="RT" else f"{condition}, L={l}"
                axis.plot(mean.index,mean.values,color=colors.get(int(l),"0.4"),linestyle=":" if condition=="RT" else "-",label=label)
            axis.set_xscale("log",base=2);axis.set_yscale("log",base=2);axis.grid(alpha=.25);axis.set_title(f"segment {seg}");axis.set_xlabel("H-update lag" if clock=="h" else "L-update lag")
        axes[0].set_ylabel("per-coordinate MSD");axes[-1].legend(fontsize=6);figure.suptitle(title);figure.tight_layout();figure.savefig(output/f"{clock}_clock_{state}_msd.png",dpi=200);figure.savefig(output/f"{clock}_clock_{state}_msd.pdf");plt.close(figure)
    if not e.empty:
        metrics=["h_jump_norm","l_block_path","l_first_quarter_step","l_last_quarter_step","h_to_l_first_alignment"]
        figure,axes=plt.subplots(1,len(metrics),figsize=(18,3.5))
        for axis,metric in zip(axes,metrics):
            q=e[e.metric==metric]
            for (condition,l),g in q.groupby(["condition","eval_l"]):
                y=g.groupby("segment")["mean"].mean(); axis.plot(y.index,y.values,marker="o",color=colors.get(int(l),"0.4"),label="RT" if condition=="RT" else f"{condition}, L={l}")
            axis.set_title(metric);axis.set_xlabel("H-clock segment");axis.grid(alpha=.25)
        axes[-1].legend(fontsize=6);figure.suptitle("H jump and L block execution");figure.tight_layout();figure.savefig(output/"h_to_l_execution.png",dpi=200);figure.savefig(output/"h_to_l_execution.pdf");plt.close(figure)
    phase=pd.DataFrame(phase_rows)
    if not phase.empty:
        figure,axes=plt.subplots(1,4,figsize=(15,3.4),sharey=True)
        for seg,axis in enumerate(axes,1):
            q=phase[phase.segment==seg]
            for (condition,l),g in q.groupby(["condition","eval_l"]):
                y=g.groupby("phase_center")["mean"].mean(); axis.plot(y.index,y.values,color=colors.get(int(l),"0.4"),label="RT" if condition=="RT" else f"{condition}, L={l}")
            axis.set_title(f"H segment {seg}");axis.set_xlabel("relative L-block phase");axis.set_yscale("log",base=2);axis.grid(alpha=.25)
        axes[0].set_ylabel("mean L step norm");axes[-1].legend(fontsize=6);figure.suptitle("L execution within each H-guided block");figure.tight_layout();figure.savefig(output/"within_block_l_phase.png",dpi=200);figure.savefig(output/"within_block_l_phase.pdf");plt.close(figure)


def seed_summary(rows: list[dict[str, object]], fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = {}
    for row in rows: grouped.setdefault(tuple(row[field] for field in fields), []).append(float(row["mean"]))
    return [{**dict(zip(fields, key)), "seed_mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)) if len(values)>1 else 0.0, "seeds": len(values)} for key, values in sorted(grouped.items())]


def finalize(args: Any) -> None:
    msd, execution, phase_rows=summarize(args.output_dir,args.bootstrap_replicates,args.sample_seed)
    atomic_csv(args.output_dir/"clock_msd_puzzle_bootstrap.csv",msd,MSD_FIELDS);atomic_csv(args.output_dir/"execution_puzzle_bootstrap.csv",execution,EXEC_FIELDS);atomic_csv(args.output_dir/"within_block_l_phase.csv",phase_rows,PHASE_FIELDS)
    atomic_csv(args.output_dir/"clock_msd_seed_summary.csv",seed_summary(msd,["condition","eval_l","clock","state","segment","lag"]),SEED_MSD_FIELDS)
    atomic_csv(args.output_dir/"execution_seed_summary.csv",seed_summary(execution,["condition","eval_l","segment","metric"]),SEED_EXEC_FIELDS)
    make_figures(args.output_dir,msd,execution,phase_rows)
    (args.output_dir/"analysis_metadata.json").write_text(json.dumps({"profile":"core-h-l-clock","total_h_updates":TOTAL_H_UPDATES,"burn_in_h_updates":0,"segments":4,"h_lags":H_LAGS.tolist(),"l_lags":L_LAGS.tolist(),"samples":args.samples,"scheme":SCHEME},indent=2)+"\n")
    (args.output_dir/"README.md").write_text("# Core H-clock / L-clock dynamics\n\nAll 160 H updates are analysed; there is no burn-in. Four H-clock segments each contain 40 H updates. H-clock MSD samples only after H updates. L-clock MSD samples every lower L update. L block execution metrics are aggregated after each H guidance block.\n")


def merge(args: Any) -> None:
    workers=args.merge_from; hashes=[(Path(x)/"sample_manifest.sha256").read_text().strip() for x in workers]
    if len(set(hashes))!=1: raise ValueError("Worker sample manifests differ.")
    rows=[]; selections={}
    for worker in workers:
        rows.extend(read_csv(Path(worker)/"rollout_metadata.csv")); selections.update({(r["condition"],int(r["seed"])):r for r in read_csv(Path(worker)/"best_checkpoints.csv")})
        for source in (Path(worker)/"per_puzzle").glob("*.npz"):
            target=args.output_dir/"per_puzzle"/source.name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if not filecmp.cmp(source,target,shallow=False): raise ValueError(f"Conflicting {target.name}")
            else: shutil.copy2(source,target)
    expected={u.key for u in args.all_units}; actual={(r["condition"],int(r["seed"]),int(r["eval_l"])) for r in rows}
    if actual!=expected: raise ValueError(f"Incomplete merge: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for r in rows:r["per_puzzle_file"]=str(args.output_dir/"per_puzzle"/Path(r["per_puzzle_file"]).name)
    atomic_csv(args.output_dir/"rollout_metadata.csv",rows,METADATA_FIELDS);atomic_csv(args.output_dir/"best_checkpoints.csv",list(selections.values()),SELECTION_FIELDS);shutil.copy2(Path(workers[0])/"sample_manifest.csv",args.output_dir/"sample_manifest.csv");shutil.copy2(Path(workers[0])/"sample_manifest.sha256",args.output_dir/"sample_manifest.sha256");finalize(args)


def main_h_l_clock(args: Any) -> None:
    args.all_units=units(core_runs(args.checkpoints_root,args.core_seeds),args.core_seeds)
    if args.merge_from: merge(args);return
    assigned=args.all_units[args.shard_index::args.num_shards] if args.shard_index is not None else args.all_units
    selected_rows=selected(args.reference_best_checkpoints,args.all_units); cfg=assigned[0].run.config
    loader,metadata=load_module(f"dataset.{cfg.data.name}@create_dataloader")("test_hard",cfg.local_batch_size,rank=0,world_size=1,**data_kwargs(cfg)); fixed_x,indices=fixed_random_samples(loader,args.samples,args.sample_seed);manifest=write_manifest(args.output_dir,indices,args.sample_seed)
    work=sum((len(H_LAGS)+len(L_LAGS if u.eval_l>1 else L_LAGS[L_LAGS<40]))*math.ceil(len(fixed_x)/args.rollout_batch_size) for u in assigned)
    progress=tqdm(total=work,desc="Core H/L-clock paired rollouts",unit="lag-batch"); rows=read_csv(args.output_dir/"rollout_metadata.csv");done={(r["condition"],int(r["seed"]),int(r["eval_l"])) for r in rows}
    for unit in assigned:
        progress.set_postfix_str(unit.label)
        if unit.key in done: continue
        row=collect_unit(unit,selected_rows[(unit.run.condition,unit.run.seed)],fixed_x,metadata,args,manifest,progress);rows=[r for r in rows if (r["condition"],int(r["seed"]),int(r["eval_l"]))!=unit.key]+[row];atomic_csv(args.output_dir/"rollout_metadata.csv",rows,METADATA_FIELDS);atomic_csv(args.output_dir/"best_checkpoints.csv",list(selected_rows.values()),SELECTION_FIELDS)
    progress.close()
    if args.num_shards==1:finalize(args)
