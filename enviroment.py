import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI, OpenAIError

# --------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------CLI ARGUMENTS --------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiDAR/vision change‑analysis pipeline")
    p.add_argument("--prompts", type=Path, default=Path("environment_prompts.json"),
                   help="Path to prompts.json file")
    p.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"),
                   help="OpenAI API key (overrides environment variable)")
    p.add_argument("--data-root", type=Path, default=Path("."),
                   help="Root directory that holds bag folders (hou-*)")
    p.add_argument("--bags", nargs="*", default=None,
                   help="Specific bag folders to process (default: all hou-*)")
    p.add_argument("--framework", choices=["zero-shot", "cot", "self-refine", "full"],
                   default="full", help="Prompting framework to use")
    return p.parse_args()

ARGS = parse_args()

# --------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------GLOBALS & INITIALISATION ---------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------
ROOT_DIR: Path = ARGS.data_root.resolve()
PROMPTS_FILE: Path = ARGS.prompts.resolve()
FRAMEWORK: str = ARGS.framework  # single source of truth downstream

with open(PROMPTS_FILE, "r", encoding="utf-8") as fp:
    PROMPTS: Dict[str, str] = json.load(fp)

client = OpenAI()

# --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- HELPERS --------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

def get_prompt(key: str) -> str:
    try:
        return PROMPTS[key]
    except KeyError as exc:
        raise KeyError(f"Prompt '{key}' missing in {PROMPTS_FILE}") from exc


def chat_completion(model: str, messages: List[Dict[str, Any]]) -> str:
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI call failed: {exc}") from exc


def encode_image(img_path: Path) -> str:
    with img_path.open("rb") as img:
        return base64.b64encode(img.read()).decode()

# --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- MODEL WRAPPERS (UNCHANGED LOGIC) -------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

def zero_shot(prev_vis: str, curr_vis: str, prev_lidar: str, curr_lidar: str) -> str:
    return chat_completion(
        "gpt-4o",
        [
            {"role": "system", "content": get_prompt("zero_shot_system")},
            {"role": "user", "content": get_prompt("zero_shot_user").format(
                prev_vis=prev_vis, curr_vis=curr_vis, prev_lidar=prev_lidar, curr_lidar=curr_lidar)}
        ],
    )


def cot(prev_vis: str, curr_vis: str, prev_lidar: str, curr_lidar: str) -> str:
    return chat_completion(
        "gpt-4o",
        [
            {"role": "system", "content": get_prompt("cot_system")},
            {"role": "user", "content": get_prompt("cot_user").format(
                prev_vis=prev_vis, curr_vis=curr_vis, prev_lidar=prev_lidar, curr_lidar=curr_lidar)}
        ],
    )


def object_analysis(b64_img: str) -> str:
    return chat_completion(
        "gpt-4o-mini",
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": get_prompt("object_analysis_text")},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ],
        }],
    )


# DEMO visual change helper ------------------------------

def visual_changes(prev_vis: str, curr_vis: str) -> str:
    return chat_completion(
        "gpt-4o",
        [
            {"role": "system", "content": get_prompt("changes_system")},
            {"role": "user", "content": get_prompt("changes_user").format(prev=prev_vis, curr=curr_vis)}
        ],
    )

# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- PROCESSING LOGIC -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

def ensure_cols(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col not in df.columns:
            df[col] = ""

def process_bag(bag_path: Path):
    csv_path = bag_path / "bag_information" / "object_descriptions.csv"
    if not csv_path.exists():
        print(f"❌ {csv_path.relative_to(ROOT_DIR)} missing – skipping.")
        return

    print(f"🔍 {csv_path.relative_to(ROOT_DIR)}")
    df = pd.read_csv(csv_path)

    ensure_cols(df, [
        "object_analysis_output", "changes_analysis_output", "lidar_changes_output",
        "cause1_analysis_output", "cause2_analysis_output", "environment_aggregation",
        "zero_shot_environment_analysis", "self_refine_environment_output", "CoT_output",
        "aggregated_driving_environment",
    ])

    # PASS 1 – object detection (always needed)
    for idx, row in df.iterrows():
        if row["object_analysis_output"]:
            continue
        img_name = row.get("env_picture_title", "")
        img_path = bag_path / "img" / img_name
        if not img_path.exists():
            continue
        df.at[idx, "object_analysis_output"] = object_analysis(encode_image(img_path))
        df.to_csv(csv_path, index=False)

    # PASS 2 – depending on framework ---------------------------------------
    for idx in range(1, len(df)):
        prev_idx = idx - 1
        prev_vis, curr_vis = df.at[prev_idx, "object_analysis_output"], df.at[idx, "object_analysis_output"]
        prev_lidar, curr_lidar = df.at[prev_idx, "vehicle_expert_output"], df.at[idx, "vehicle_expert_output"]
        if not curr_vis:
            continue

        # ---------------- choose framework tasks ---------------------------
        if FRAMEWORK in ("cot", "full") and not df.at[idx, "CoT_output"]:
            df.at[idx, "CoT_output"] = cot(prev_vis, curr_vis, prev_lidar, curr_lidar)

        if FRAMEWORK in ("self-refine", "full") and not df.at[idx, "self_refine_environment_output"]:
            df.at[idx, "self_refine_environment_output"] = self_refine_objects(prev_vis, curr_vis, prev_lidar, curr_lidar)

        if FRAMEWORK in ("cot", "full"):
            # Visual + lidar change pipelines (needed by cot/full)
            if not df.at[idx, "changes_analysis_output"]:
                df.at[idx, "changes_analysis_output"] = visual_changes(prev_vis, curr_vis)
            if not df.at[idx, "lidar_changes_output"]:
                df.at[idx, "lidar_changes_output"] = lidar_changes_analysis(prev_lidar, curr_lidar)
            if not df.at[idx, "cause1_analysis_output"]:
                df.at[idx, "cause1_analysis_output"] = cause1_analysis(
                    df.at[idx, "changes_analysis_output"], df.at[idx, "lidar_changes_output"]
                )
            if not df.at[idx, "cause2_analysis_output"]:
                df.at[idx, "cause2_analysis_output"] = cause2_analysis(
                    df.at[idx, "changes_analysis_output"], df.at[idx, "lidar_changes_output"]
                )
            if not df.at[idx, "aggregated_driving_environment"]:
                df.at[idx, "aggregated_driving_environment"] = cause_aggregation(
                    df.at[idx, "cause1_analysis_output"], df.at[idx, "cause2_analysis_output"]
                )

        if FRAMEWORK in ("zero-shot", "full") and not df.at[idx, "zero_shot_environment_analysis"]:
            df.at[idx, "zero_shot_environment_analysis"] = zero_shot(prev_vis, curr_vis, prev_lidar, curr_lidar)

        # Minimal yes/no aggregation to show something for every framework
        if not df.at[idx, "environment_aggregation"]:
            df.at[idx, "environment_aggregation"] = aggregation(df.at[idx, "changes_analysis_output"] or curr_vis)

        df.to_csv(csv_path, index=False)
        print(f"DONE for Row {idx+1} done (framework: {FRAMEWORK})!")

if __name__ == "__main__":
    bag_dirs = ARGS.bags if ARGS.bags else [d.name for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("hou-")]
    for bag in bag_dirs:
        process_bag(ROOT_DIR / bag)
