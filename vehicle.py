import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from openai import OpenAI, OpenAIError

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vehicle sensor abnormality detection pipeline")
    p.add_argument("--prompts", type=Path, default=Path("vehicle_prompts.json"),
                   help="Path to prompts JSON file")
    p.add_argument("--data-root", type=Path, default=Path("."),
                   help="Root directory containing bag folders (hou-*)")
    p.add_argument("--bags", nargs="*", default=None,
                   help="Specific bag folders to process (default: all hou-*)")
    p.add_argument("--framework", choices=["zero-shot", "cot", "self-refine", "full"],
                   default="full", help="Prompting framework to use")
    p.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"),
                   help="OpenAI API key (overrides env var)")
    return p.parse_args()

ARGS = parse_args()

os.environ["OPENAI_API_KEY"] = ARGS.api_key or ""
if not os.environ["OPENAI_API_KEY"]:
    raise RuntimeError("OpenAI API key not provided via --api-key or env var")

ROOT_DIR: Path = ARGS.data_root.resolve()
PROMPTS_FILE: Path = ARGS.prompts.resolve()
FRAMEWORK: str = ARGS.framework

with open(PROMPTS_FILE, "r", encoding="utf-8") as fp:
    raw_prompts: Dict[str, Any] = json.load(fp)

if "vehicle" in raw_prompts and isinstance(raw_prompts["vehicle"], dict):
    PROMPTS: Dict[str, str] = raw_prompts["vehicle"]
else:
    PROMPTS = raw_prompts

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def gprompt(key: str) -> str:
    try:
        return PROMPTS[key]
    except KeyError as exc:
        raise KeyError(f"Prompt '{key}' missing under 'vehicle' in {PROMPTS_FILE}") from exc


def chat(model: str, messages: List[Dict[str, Any]]) -> str:
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI error: {exc}") from exc

def lidar_expert(desc: str) -> str:
    return chat("gpt-4o", [
        {"role": "system", "content": gprompt("lidar_expert_system")},
        {"role": "user",   "content": desc},
    ])


def abnormal_analysis(lidar_out: str, visual_desc: str) -> str:
    return chat("gpt-4o", [
        {"role": "system", "content": gprompt("abnormal_system")},
        {"role": "user",   "content": gprompt("abnormal_user").format(lidar=lidar_out, visual=visual_desc)},
    ])


def zero_shot(desc: str, visual_desc: str) -> str:
    return chat("gpt-4o", [
        {"role": "system", "content": gprompt("zero_shot_system")},
        {"role": "user",   "content": gprompt("zero_shot_user").format(lidar=desc, visual=visual_desc)},
    ])


def cot(desc: str, visual_desc: str) -> str:
    return chat("gpt-4o", [
        {"role": "system", "content": gprompt("cot_system")},
        {"role": "user",   "content": gprompt("cot_user").format(lidar=desc, visual=visual_desc)},
    ])


def aggregation(lidar_out: str, visual_desc: str, abnormal_out: str) -> str:
    return chat("gpt-4o", [
        {"role": "system", "content": gprompt("agg_system")},
        {"role": "user",   "content": gprompt("agg_user").format(lidar=lidar_out, visual=visual_desc, abnormal=abnormal_out)},
    ])


def self_refine_agent(desc: str, visual_desc: str) -> str:
    return chat("gpt-4o", [
        {"role": "system", "content": gprompt("sr_system")},
        {"role": "user",   "content": gprompt("sr_user").format(lidar=desc, visual=visual_desc)},
    ])


def ensure_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = ""


def process_bag(bag: Path):
    csv_path = bag / "bag_information/object_descriptions.csv"
    if not csv_path.exists():
        print(f"❌ {csv_path.relative_to(ROOT_DIR)} missing – skip")
        return

    df = pd.read_csv(csv_path)
    ensure_cols(df, [
        "vehicle_expert_output", "vehicle_abnormal_analysis_output",
        "vehicle_aggregation_output", "vehicle_zero_shot_analysis",
        "vehicle_self_refine_analysis", "vehicle_cot_analysis",
    ])

    for idx, row in df.iterrows():
        lidar_desc = row.get("object_description", "")
        visual_desc = row.get("description", "")

        # zero‑shot ----------------------------------------------------
        if FRAMEWORK in ("zero-shot", "full") and not row["vehicle_zero_shot_analysis"]:
            df.at[idx, "vehicle_zero_shot_analysis"] = zero_shot(lidar_desc, visual_desc)

        # LiDAR expert + abnormal -------------------------------------
        if FRAMEWORK in ("cot", "self-refine", "full"):
            if not row["vehicle_expert_output"]:
                df.at[idx, "vehicle_expert_output"] = lidar_expert(lidar_desc)
            if not row["vehicle_abnormal_analysis_output"]:
                df.at[idx, "vehicle_abnormal_analysis_output"] = abnormal_analysis(
                    df.at[idx, "vehicle_expert_output"], visual_desc
                )

        # aggregation + CoT -------------------------------------------
        if FRAMEWORK in ("cot", "full"):
            if not row["vehicle_aggregation_output"]:
                df.at[idx, "vehicle_aggregation_output"] = aggregation(
                    df.at[idx, "vehicle_expert_output"], visual_desc,
                    df.at[idx, "vehicle_abnormal_analysis_output"]
                )
            if not row["vehicle_cot_analysis"]:
                df.at[idx, "vehicle_cot_analysis"] = cot(lidar_desc, visual_desc)

        # self‑refine ---------------------------------------------------
        if FRAMEWORK in ("self-refine", "full") and not row["vehicle_self_refine_analysis"]:
            df.at[idx, "vehicle_self_refine_analysis"] = self_refine_agent(lidar_desc, visual_desc)

        df.to_csv(csv_path, index=False)
        print(f"✅ {csv_path.name} row {idx+1} done ({FRAMEWORK}).")

if __name__ == "__main__":
    bags = ARGS.bags if ARGS.bags else [d.name for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("hou-")]
    for b in bags:
        process_bag(ROOT_DIR / b)