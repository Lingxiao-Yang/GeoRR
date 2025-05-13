import csv
import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from processing import FilterHR  # ← your Phase‑III implementation

# --------------------------------------------------------------------------- #
#                      tweakable parameters                                   #
# --------------------------------------------------------------------------- #
# TEST_FILE     = "./Data/environment_1_60s_20kHz.txt"   # input path
TEST_FILE     = "./Data/test1.txt"   # input path
OUTPUT_ROOT   = "./Output"           # parent of all run folders
SAMPLE_RATE   = 20000                  # Hz
VISUALIZE     = True                 # save PNGs of raw traces
NUM_CHANNELS  = 1                    # fixed by spec
# --------------------------------------------------------------------------- #


def read_intensity(path: str, n_channels: int = NUM_CHANNELS) -> np.ndarray:
    """Load `n_channels` traces from text file → shape (n_channels, T)."""
    data = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != n_channels:
                raise ValueError(
                    f"Line {line_no}: expected {n_channels} numbers, got {len(parts)}"
                )
            data.append([float(x) for x in parts])

    if not data:
        raise ValueError(f"No valid data found in {path}")

    return np.asarray(data, dtype=float).T  # → (T, C)  → transpose → (C, T)


def plot_trace(trace: np.ndarray, save_dir: Path, label: str) -> None:
    """Save a PNG of a single-channel trace."""
    frames = np.arange(trace.size)
    plt.figure(figsize=(20, 4))
    plt.plot(frames, trace, linewidth=0.8)
    plt.title(f"Raw Intensity - {label}")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir / f"{label}_intensity.png", dpi=300)
    plt.close()


def main() -> None:
    t0 = time.time()
    basename = Path(TEST_FILE).stem
    out_dir  = Path(OUTPUT_ROOT) / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- #
    #   load data                                                         #
    # ------------------------------------------------------------------- #
    intensity_all = read_intensity(TEST_FILE)
    channels     = [f"ch{i}" for i in range(NUM_CHANNELS)]
    intensity = [trace for trace in intensity_all]
    
    
    # Down sample to 10000 Hz and 20sec
    down_sample_rate = 10000
    down_sample_duration = 20
    intensity = [trace[0:SAMPLE_RATE * down_sample_duration] for trace in intensity]
    intensity = [trace[::int(SAMPLE_RATE / down_sample_rate)] for trace in intensity]

    if VISUALIZE:
        for i, trace in enumerate(intensity):
            plot_trace(trace, out_dir, channels[i])

    # ------------------------------------------------------------------- #
    #   Signal Processing                                                 #
    # ------------------------------------------------------------------- #
    print("Running Phase III (HR extraction)…")
    phase3   = FilterHR(
        intensity  = intensity,
        channels  = channels,
        sample_rate = down_sample_rate,
        output_DIR = str(out_dir),
        verbose= True
    )
    hr, hr_conf = phase3.run()
    print("Phase III done.\n")

    baseline_hr = np.sum(np.asarray(hr) * np.asarray(hr_conf)) / np.sum(hr_conf)

    # --------------------------- console summary ----------------------- #
    print(f"{'ROI':<8} | {'HR (bpm)':^10} | {'Confidence':^10}")
    print("-" * 34)
    for r, h, c in zip(channels, hr, hr_conf):
        print(f"{r:<8} | {h:^10.2f} | {c:^10.2f}")
    print(f"\nBaseline HR: {baseline_hr:.2f} bpm\n")

    print(f"Total runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
