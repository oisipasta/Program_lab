from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, iirnotch


# ============================
# 設定
# ============================

@dataclass
class Config:
    root: Path

    emg_dirname: str = "emg"
    pressure_dirname: str = "mov"
    out_dirname: str = "_analysis_output"

    # ---- EMG ----
    emg_fs_hz: float = 1000.0

    # 10-500Hz bandpass
    bandpass_low_hz: float = 10.0
    bandpass_high_hz: float = 500.0
    bandpass_order: int = 4

    # 50Hz notch
    notch_hz: float = 50.0
    notch_q: float = 30.0

    # Wavelet denoise
    wavelet: str = "db4"
    wavelet_level: int | None = None  # Noneなら自動

    # 100ms moving RMS smoothing
    rms_window_ms: float = 100.0

    # 採用区間（EMGは中央窓）
    keep_minutes_normal: int = 5
    keep_minutes_validation: int = 20
    validation_keyword: str = "validation"


# ============================
# ファイル名の解釈
# ============================

def parse_condition_from_filename(subject: str, path: Path, cfg: Config) -> dict:
    """
    例（※日付があってもなくてもOK）:
      kaori_aomuke.csv
      kaori_aomuke_20260220.csv
      kaori_yuka_aomuke.csv
      kaori_validation.csv
    """
    stem = path.stem

    # subject_ を削る
    if stem.startswith(subject + "_"):
        stem = stem[len(subject) + 1 :]

    # 末尾の日付 _YYMMDD / _YYYYMMDD があれば削る（無ければ何もしない）
    stem = re.sub(r"_(\d{6,8})$", "", stem)

    cond = stem
    low = cond.lower()

    is_validation = (cfg.validation_keyword.lower() in low)

    # yuka 含む -> 床, 含まない -> マットレス
    environment = "yuka" if ("yuka" in low) else "mattress"

    # 姿勢はユーザー指定の表記に合わせる
    posture = None
    if "aomuke" in low:
        posture = "aomuke"
    elif "utubuse" in low or "utsubuse" in low:
        posture = "utubuse"
    elif "yokomuki" in low:
        posture = "yokomuki"
    elif is_validation:
        posture = "aomuke"  # validation は仰臥位扱い

    return {
        "condition": cond,
        "environment": environment,
        "posture": posture,          # aomuke / utubuse / yokomuki
        "is_validation": is_validation,
    }


# ============================
# EMG 読込（タブ区切り / 4ch + 末尾タブ想定）
# ============================

def read_emg_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    df = df.dropna(axis=1, how="all")  # 末尾タブ由来の空列を落とす
    if df.shape[1] != 4:
        raise ValueError(f"EMG列数が4ではありません: {path.name} (cols={df.shape[1]})")
    df.columns = ["ch1", "ch2", "ch3", "ch4"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def center_window_indices_by_samples(n: int, keep_minutes: int, fs: float) -> slice:
    keep = int(round(keep_minutes * 60.0 * fs))
    if n < keep:
        raise ValueError(f"EMGデータが短すぎます: n={n} < keep={keep}")
    start = (n - keep) // 2
    end = start + keep
    return slice(start, end)


def design_bandpass(cfg: Config):
    fs = float(cfg.emg_fs_hz)
    nyq = fs / 2.0

    low = cfg.bandpass_low_hz / nyq

    # high=500Hz はNyquist(=500)に一致して設計不可なので少し下げて実装
    high_hz = min(cfg.bandpass_high_hz, nyq * 0.999)
    high = high_hz / nyq

    if not (0 < low < high < 1):
        raise ValueError(f"Bandpass範囲が不正: low={cfg.bandpass_low_hz}, high={cfg.bandpass_high_hz}, fs={fs}")

    b, a = butter(cfg.bandpass_order, [low, high], btype="bandpass")
    return b, a


def apply_bandpass(x: np.ndarray, cfg: Config) -> np.ndarray:
    b, a = design_bandpass(cfg)
    x = np.nan_to_num(x, nan=0.0).astype(float, copy=False)
    return filtfilt(b, a, x)


def apply_notch_50hz(x: np.ndarray, cfg: Config) -> np.ndarray:
    fs = float(cfg.emg_fs_hz)
    w0 = cfg.notch_hz / (fs / 2.0)
    b, a = iirnotch(w0=w0, Q=cfg.notch_q)
    return filtfilt(b, a, x)


def wavelet_denoise(x: np.ndarray, cfg: Config) -> np.ndarray:
    w = pywt.Wavelet(cfg.wavelet)
    max_level = pywt.dwt_max_level(len(x), w.dec_len)
    level = cfg.wavelet_level if cfg.wavelet_level is not None else max(1, min(6, max_level))

    coeffs = pywt.wavedec(x, w, level=level, mode="periodization")

    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
    if sigma <= 0:
        return x.copy()

    uthresh = sigma * np.sqrt(2.0 * np.log(len(x)))

    coeffs_f = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_f.append(pywt.threshold(c, value=uthresh, mode="soft"))

    y = pywt.waverec(coeffs_f, w, mode="periodization")
    return y[: len(x)]


def moving_rms(x: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    w = int(round(fs * (window_ms / 1000.0)))
    w = max(1, w)
    x2 = x.astype(np.float32, copy=False) ** 2
    kernel = np.ones(w, dtype=np.float32) / np.float32(w)
    ma = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ma, dtype=np.float32)


def emg_pipeline(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for ch in df.columns:
        x = df[ch].to_numpy(dtype=float)
        x = apply_bandpass(x, cfg)
        x = apply_notch_50hz(x, cfg)
        x = wavelet_denoise(x, cfg)
        x = moving_rms(x, cfg.emg_fs_hz, cfg.rms_window_ms)
        out[ch] = x
    return out


def analyze_emg_file(subject: str, csv_path: Path, cfg: Config, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = parse_condition_from_filename(subject, csv_path, cfg)

    raw = read_emg_csv(csv_path)
    proc = emg_pipeline(raw, cfg)

    keep_min = cfg.keep_minutes_validation if meta["is_validation"] else cfg.keep_minutes_normal
    sl = center_window_indices_by_samples(len(proc), keep_min, cfg.emg_fs_hz)
    win = proc.iloc[sl].reset_index(drop=True)

    t = np.arange(len(win), dtype=float) / float(cfg.emg_fs_hz)
    minute_bin = np.floor(t / 60.0).astype(int)

    per_min_records = []
    for m in range(keep_min):
        sel = minute_bin == m
        if not np.any(sel):
            continue
        for ch in win.columns:
            v = win.loc[sel, ch].to_numpy(dtype=float)
            per_min_records.append({
                "subject": subject,
                **meta,
                "minute": int(m),
                "channel": ch,
                "mean_rms100ms": float(np.nanmean(v)),
            })
    per_min_df = pd.DataFrame(per_min_records)

    overall_records = []
    for ch in win.columns:
        v = win[ch].to_numpy(dtype=float)
        overall_records.append({
            "subject": subject,
            **meta,
            "channel": ch,
            "mean_rms100ms": float(np.nanmean(v)),
        })
    overall_df = pd.DataFrame(overall_records)

    return per_min_df, overall_df


# ============================
# 体圧 読込（“Measuring Time,” 行から本体）
# ============================

def find_pressure_header_line(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("Measuring Time,"):
                return i
    raise ValueError(f"体圧CSVのヘッダ行が見つかりません: {path.name}")


def read_pressure_csv(path: Path) -> pd.DataFrame:
    header_i = find_pressure_header_line(path)
    df = pd.read_csv(path, skiprows=header_i, index_col=False, engine="python")
    if "Elapsed Time[s]" not in df.columns:
        raise ValueError(f"体圧CSVに 'Elapsed Time[s]' がありません: {path.name}")
    return df


def get_pressure_sensor_cols(df: pd.DataFrame) -> list[str]:
    sensor_cols = [c for c in df.columns if re.fullmatch(r"X\d+Y\d+", str(c))]
    if len(sensor_cols) != 1600:
        raise ValueError(f"体圧センサ列が1600個ではありません: found={len(sensor_cols)}")
    return sensor_cols


def analyze_pressure_file(subject: str, csv_path: Path, cfg: Config, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    体圧は先頭から keep_sec までを使用
      - 超えていればカット
      - 超えていなければある分だけで計算
    """
    meta = parse_condition_from_filename(subject, csv_path, cfg)

    df = read_pressure_csv(csv_path)
    sensor_cols = get_pressure_sensor_cols(df)

    t = pd.to_numeric(df["Elapsed Time[s]"], errors="coerce").to_numpy(dtype=float)

    keep_min = cfg.keep_minutes_validation if meta["is_validation"] else cfg.keep_minutes_normal
    keep_sec = float(keep_min * 60.0)

    mask = np.isfinite(t) & (t >= 0.0) & (t <= keep_sec)
    dfw = df.loc[mask].reset_index(drop=True)
    if dfw.empty:
        raise ValueError("体圧データが0行になりました（時間列を確認してください）。")

    tw = pd.to_numeric(dfw["Elapsed Time[s]"], errors="coerce").to_numpy(dtype=float)
    tw = tw - float(np.nanmin(tw))

    vals = dfw[sensor_cols].to_numpy(dtype=np.float32, copy=False)

    # 非ゼロ平均
    nonzero = (vals != 0) & np.isfinite(vals)
    cnt = nonzero.sum(axis=1).astype(np.float32)  # 接触面積（セル数）
    s = (vals * nonzero).sum(axis=1)

    mean_nz = np.full_like(s, np.nan, dtype=np.float32)
    ok = cnt > 0
    mean_nz[ok] = s[ok] / cnt[ok]

    minute_bin = np.floor(tw / 60.0).astype(int)

    per_min_records = []
    for m in sorted(set(minute_bin.tolist())):
        if m < 0 or m >= keep_min:
            continue
        sel = minute_bin == m
        per_min_records.append({
            "subject": subject,
            **meta,
            "minute": int(m),
            "mean_pressure_nonzero": float(np.nanmean(mean_nz[sel])),
            "contact_area_cells": float(np.nanmean(cnt[sel])),
        })
    per_min_df = pd.DataFrame(per_min_records)

    overall_df = pd.DataFrame([{
        "subject": subject,
        **meta,
        "mean_pressure_nonzero": float(np.nanmean(mean_nz)),
        "contact_area_cells": float(np.nanmean(cnt)),
        "used_seconds": float(np.nanmax(tw)),
    }])

    return per_min_df, overall_df


# ============================
# ディレクトリ走査・進捗
# ============================

def list_subject_dirs(base: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not base.exists():
        return out
    for d in base.iterdir():
        if d.is_dir() and d.name.endswith(suffix):
            subject = d.name[: -len(suffix)]
            out[subject] = d
    return out


def progress_print(done: int, total: int):
    pct = (done / total * 100.0) if total > 0 else 100.0
    print(f"\rProgress: {pct:6.2f}% ({done}/{total})", end="", flush=True)

def _cond_order_key(cond: str) -> int:
    order = [
        "aomuke", "utubuse", "yokomuki",
        "validation",
        "yuka_aomuke", "yuka_utubuse", "yuka_yokomuki",
    ]
    return order.index(cond) if cond in order else 999


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    # txt用の簡易テーブル整形
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    line = "  " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "  " + "-+-".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for r in rows:
        out.append("  " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return out

# ============================
# TXT出力（姿勢表記を aomuke/utubuse/yokomuki に）
# ============================

def _env_jp(env: str) -> str:
    return "床" if env == "yuka" else "マットレス"

def _posture_jp(pos: str | None) -> str:
    if pos == "aomuke":
        return "仰臥位（aomuke）"
    if pos == "utubuse":
        return "伏臥位（utubuse）"
    if pos == "yokomuki":
        return "側臥位（yokomuki）"
    return "不明"

def _bool_jp(x: bool) -> str:
    return "はい" if x else "いいえ"


def write_subject_txt(out_root: Path, subject: str,
                      emg_overall: pd.DataFrame, emg_per_min: pd.DataFrame,
                      pr_overall: pd.DataFrame, pr_per_min: pd.DataFrame,
                      cfg: Config):
    lines = []
    lines.append(f"被験者名：{subject}")
    lines.append("=" * 70)

    lines.append("\n【筋電（EMG）解析設定】")
    lines.append(f"  サンプリング周波数：{cfg.emg_fs_hz} Hz")
    lines.append(f"  バンドパス：{cfg.bandpass_low_hz}–{cfg.bandpass_high_hz} Hz")
    lines.append(f"  ノッチ：{cfg.notch_hz} Hz（Q={cfg.notch_q}）")
    lines.append(f"  Waveletノイズ除去：{cfg.wavelet}（レベル：{cfg.wavelet_level if cfg.wavelet_level is not None else '自動'}）")
    lines.append(f"  平滑化：移動RMS {cfg.rms_window_ms} ms")
    lines.append("  出力指標：mean_rms100ms（100ms移動RMS包絡の平均）")

    # =======================
    # EMG：全体平均 + 各分平均
    # =======================
    if emg_overall.empty:
        lines.append("\n【筋電（EMG）結果】\n  （結果なし）")
    else:
        lines.append("\n【筋電（EMG）結果】")
        for cond, g_over in sorted(emg_overall.groupby("condition"), key=lambda x: _cond_order_key(x[0])):
            env = _env_jp(g_over["environment"].iloc[0])
            pos = _posture_jp(g_over["posture"].iloc[0])
            is_val = _bool_jp(bool(g_over["is_validation"].iloc[0]))

            lines.append(f"\n■ 条件：{cond}")
            lines.append(f"  環境：{env} / 姿勢：{pos} / validation：{is_val}")

            # 全体平均（ch1-4）
            lines.append("  【測定全体の平均】")
            ch_rows = []
            for _, r in g_over.sort_values("channel").iterrows():
                ch_rows.append([str(r["channel"]), f"{r['mean_rms100ms']:.6g}"])
            lines += _format_table(["ch", "mean_rms100ms"], ch_rows)

            # 部位差（最大-最小）
            vals = g_over.set_index("channel")["mean_rms100ms"].astype(float)
            ch_max = vals.idxmax()
            ch_min = vals.idxmin()
            diff = float(vals[ch_max] - vals[ch_min])
            lines.append(f"  【部位差（参考）】最大={ch_max} 最小={ch_min} 差={diff:.6g}")

            # 各分平均（トレンド）
            g_pm = emg_per_min[emg_per_min["condition"] == cond] if not emg_per_min.empty else pd.DataFrame()
            if g_pm.empty:
                lines.append("  【各分の平均（トレンド）】（データなし）")
            else:
                pivot = g_pm.pivot(index="minute", columns="channel", values="mean_rms100ms").sort_index()
                for ch in ["ch1", "ch2", "ch3", "ch4"]:
                    if ch not in pivot.columns:
                        pivot[ch] = np.nan
                pivot = pivot[["ch1", "ch2", "ch3", "ch4"]]

                rows = []
                for m in pivot.index.tolist():
                    row = [str(int(m))] + [
                        f"{pivot.loc[m, ch]:.6g}" if np.isfinite(pivot.loc[m, ch]) else "-"
                        for ch in pivot.columns
                    ]
                    rows.append(row)

                lines.append("  【各分の平均（トレンド）】")
                lines += _format_table(["minute", "ch1", "ch2", "ch3", "ch4"], rows)

    # =======================
    # 体圧：全体平均 + 各分平均
    # =======================
    if pr_overall.empty:
        lines.append("\n\n【体圧結果】\n  （結果なし）")
    else:
        lines.append("\n\n【体圧結果】")
        for cond, g_over in sorted(pr_overall.groupby("condition"), key=lambda x: _cond_order_key(x[0])):
            env = _env_jp(g_over["environment"].iloc[0])
            pos = _posture_jp(g_over["posture"].iloc[0])
            is_val = _bool_jp(bool(g_over["is_validation"].iloc[0]))

            mp = float(g_over["mean_pressure_nonzero"].iloc[0])
            ca = float(g_over["contact_area_cells"].iloc[0])
            used = float(g_over["used_seconds"].iloc[0]) if "used_seconds" in g_over.columns else float("nan")

            lines.append(f"\n■ 条件：{cond}")
            lines.append(f"  環境：{env} / 姿勢：{pos} / validation：{is_val}")

            lines.append("  【測定全体の平均】")
            lines.append(f"    全体平均体圧（非ゼロ平均）：{mp:.6g}")
            lines.append(f"    全体平均接触セル数：{ca:.3f}")
            lines.append(f"    使用時間（秒）：{used:.3f}")

            g_pm = pr_per_min[pr_per_min["condition"] == cond] if not pr_per_min.empty else pd.DataFrame()
            if g_pm.empty:
                lines.append("  【各分の平均】（データなし）")
            else:
                g_pm = g_pm.sort_values("minute")
                rows = []
                for _, r in g_pm.iterrows():
                    rows.append([
                        str(int(r["minute"])),
                        f"{float(r['mean_pressure_nonzero']):.6g}",
                        f"{float(r['contact_area_cells']):.3f}",
                    ])
                lines.append("  【各分の平均】")
                lines += _format_table(["minute", "平均体圧（非ゼロ）", "接触セル数"], rows)

    txt_path = out_root / f"{subject}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def write_all_txt(out_root: Path, subjects: list[str]):
    lines = []
    lines.append("All subjects summary")
    lines.append("=" * 70)
    for s in subjects:
        lines.append(f"- {s}.txt")
    (out_root / "summary_all.txt").write_text("\n".join(lines), encoding="utf-8")


# ============================
# main
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="PROGRAM_STUDY/mattress_summary",
                    help="mattress_summary のパス（emg/ と mov/ を含む）")
    args = ap.parse_args()

    cfg = Config(root=Path(args.root))

    emg_root = cfg.root / cfg.emg_dirname
    pr_root = cfg.root / cfg.pressure_dirname

    emg_subjects = list_subject_dirs(emg_root, "_emg")
    pr_subjects = list_subject_dirs(pr_root, "_mov")

    subjects = sorted(set(emg_subjects.keys()) | set(pr_subjects.keys()))

    out_root = cfg.root / cfg.out_dirname
    out_root.mkdir(parents=True, exist_ok=True)

    emg_files = []
    for subject, sdir in emg_subjects.items():
        emg_files += [(subject, p) for p in sorted(sdir.glob("*.csv"))]

    pr_files = []
    for subject, sdir in pr_subjects.items():
        pr_files += [(subject, p) for p in sorted(sdir.glob("*.csv"))]

    total = len(emg_files) + len(pr_files)
    done = 0
    progress_print(done, total)

    all_emg_overall = []
    all_emg_per_min = []
    all_pr_overall = []
    all_pr_per_min = []


    for subject, p in emg_files:
        try:
            subject_out = out_root / subject
            per_min, overall = analyze_emg_file(subject, p, cfg, subject_out)
            all_emg_per_min.append(per_min)
            all_emg_overall.append(overall)
        except Exception as e:
            print(f"\n[EMG] skip {p.name}: {e}")
        done += 1
        progress_print(done, total)

    for subject, p in pr_files:
        try:
            subject_out = out_root / subject
            per_min, overall = analyze_pressure_file(subject, p, cfg, subject_out)
            all_pr_per_min.append(per_min)
            all_pr_overall.append(overall)
        except Exception as e:
            print(f"\n[Pressure] skip {p.name}: {e}")
        done += 1
        progress_print(done, total)

    print()

    emg_overall_df = pd.concat(all_emg_overall, ignore_index=True) if all_emg_overall else pd.DataFrame()
    emg_per_min_df = pd.concat(all_emg_per_min, ignore_index=True) if all_emg_per_min else pd.DataFrame()
    pr_overall_df = pd.concat(all_pr_overall, ignore_index=True) if all_pr_overall else pd.DataFrame()
    pr_per_min_df = pd.concat(all_pr_per_min, ignore_index=True) if all_pr_per_min else pd.DataFrame()


    for subject in subjects:
        e = emg_overall_df[emg_overall_df["subject"] == subject] if not emg_overall_df.empty else pd.DataFrame()
        r = pr_overall_df[pr_overall_df["subject"] == subject] if not pr_overall_df.empty else pd.DataFrame()
        epm = emg_per_min_df[emg_per_min_df["subject"] == subject] if not emg_per_min_df.empty else pd.DataFrame()
        rpm = pr_per_min_df[pr_per_min_df["subject"] == subject] if not pr_per_min_df.empty else pd.DataFrame()
        write_subject_txt(out_root, subject, e, epm, r, rpm, cfg)

    write_all_txt(out_root, subjects)

    print(f"Done. Output: {out_root}")


if __name__ == "__main__":
    main()