import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

_DEF_SKIP = {"postprocess"}
_POST_PROCESS_STAGES = (
  "metrical_prep",
  "metrical_dbn",
  "functional_probs",
  "functional_local_maxima",
  "functional_boundaries",
)


def _collapse_post_process(rows: list[dict]) -> list[dict]:
  stages = set(_POST_PROCESS_STAGES)
  selected = [r for r in rows if r.get("stage") in stages]
  if not selected:
    return rows

  start = min(float(r.get("start", 0.0)) for r in selected)
  end = max(float(r.get("end", 0.0)) for r in selected)
  duration = end - start
  track = None
  for r in selected:
    if r.get("track"):
      track = r.get("track")
      break

  first_idx = None
  for idx, r in enumerate(rows):
    if r.get("stage") in stages:
      first_idx = idx
      break

  collapsed = []
  inserted = False
  for idx, r in enumerate(rows):
    if r.get("stage") in stages:
      if idx == first_idx and not inserted:
        collapsed.append({
          "stage": "post_process",
          "start": start,
          "end": end,
          "duration": duration,
          "track": track,
        })
        inserted = True
      continue
    collapsed.append(r)

  return collapsed


def visualize_timings(
  timings_path: Path,
  out_path: Path,
  title: Optional[str] = None,
  skip_stages: Optional[Iterable[str]] = None,
) -> Path:
  timings_path = Path(timings_path)
  out_path = Path(out_path)
  if not timings_path.is_file():
    raise FileNotFoundError(f"timings file not found: {timings_path}")

  rows = []
  for line in timings_path.read_text().splitlines():
    if not line.strip():
      continue
    rows.append(json.loads(line))

  if not rows:
    raise ValueError("timings file is empty")

  rows = _collapse_post_process(rows)

  skip = set(_DEF_SKIP if skip_stages is None else skip_stages)
  rows = [r for r in rows if r.get("stage") not in skip]
  if not rows:
    raise ValueError("no timing rows left after filtering")

  rows = sorted(rows, key=lambda r: float(r.get("start", 0.0)))
  t0 = float(rows[0].get("start", 0.0))

  track_names = {Path(r["track"]).name for r in rows if r.get("track")}
  if len(track_names) > 1:
    raise ValueError("timings visualization currently supports single-track runs only")
  track_name = next(iter(track_names), None)

  labels = []
  starts = []
  durations = []
  ends = []
  for r in rows:
    stage = r.get("stage", "unknown")
    stage = stage.replace("_", " ")
    start = float(r.get("start", 0.0)) - t0
    end = float(r.get("end", 0.0)) - t0
    dur = float(r.get("duration", max(end - start, 0.0)))
    labels.append(stage)
    starts.append(start)
    durations.append(dur)
    ends.append(end)

  total_duration = max(ends) if ends else 0.0

  fig_height = max(2.0, 0.4 * len(labels))
  fig, ax = plt.subplots(figsize=(12, fig_height))
  y_positions = list(range(len(labels)))

  cmap = plt.get_cmap("viridis")
  colors = [cmap(i / max(1, len(labels) - 1)) for i in range(len(labels))]

  ax.barh(y_positions, durations, left=starts, color=colors)
  ax.set_yticks(y_positions, labels)
  ax.invert_yaxis()
  ax.set_xlabel("Time (s)")

  main_title = title or "All-in-One Timing Breakdown"
  if track_name:
    ax.set_title(f"{main_title} — {track_name}")
  else:
    ax.set_title(main_title)

  for label in ax.get_yticklabels():
    label.set_fontweight("bold")

  for y, end, dur in zip(y_positions, ends, durations):
    ax.text(end + 0.01, y, f"{dur:.3f}s", va="center", fontsize=8)

  ax.grid(True, axis="x", linestyle="--", alpha=0.3)
  for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

  ax.text(
    0.99,
    -0.08,
    f"Total: {total_duration:.3f}s",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
  )

  fig.tight_layout()

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  return out_path
