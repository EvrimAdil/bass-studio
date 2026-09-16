import base64
import json
import os
import shutil
import tempfile
from pathlib import Path

import demucs.api
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import torch
import yt_dlp

try:
    import essentia.standard as es
except Exception:
    es = None

try:
    from audio_separator.separator import Separator as UVRSeparator
except Exception:
    UVRSeparator = None

try:
    from basic_pitch.inference import predict as basic_pitch_predict
except Exception:
    basic_pitch_predict = None

st.set_page_config(page_title="Bass Studio", page_icon="🎸", layout="wide", initial_sidebar_state="collapsed")

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
NOTE_TR = {"C": "Do", "C#": "Do#", "D": "Re", "D#": "Re#", "E": "Mi", "F": "Fa", "F#": "Fa#", "G": "Sol", "G#": "Sol#", "A": "La", "A#": "La#", "B": "Si"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650;800&family=IBM+Plex+Mono:wght@500;700&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { background:#070a12; color:#eef2ff; font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif; }
.main .block-container { max-width:1220px; padding:1rem 1.25rem 2.5rem; }
#MainMenu, header, footer { visibility:hidden; }
.app-hero { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:10px 0 16px; border-bottom:1px solid rgba(255,255,255,.075); margin-bottom:16px; }
.brand { font-size:1.55rem; font-weight:850; letter-spacing:-.045em; }
.brand span { background:linear-gradient(120deg,#8b5cf6,#22d3ee 55%,#34d399); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.pill { color:#a7b0c8; border:1px solid rgba(255,255,255,.1); border-radius:999px; padding:7px 11px; font:700 .72rem 'IBM Plex Mono',monospace; background:rgba(255,255,255,.035); }
.card { border:1px solid rgba(255,255,255,.075); background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.025)); border-radius:18px; padding:15px; box-shadow:0 20px 50px rgba(0,0,0,.25); }
.small-note { color:#94a3b8; font-size:.84rem; margin-top:8px; }
.stButton > button { border-radius:14px !important; font-weight:760 !important; }
.stTextInput input, .stSelectbox div[data-baseweb="select"] { border-radius:14px !important; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def audio_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext in [".m4a", ".aac"]:
        return "audio/aac"
    return "audio/wav"


def download_youtube_audio(query_or_url: str, output_dir: str):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "yt_audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "256"}],
        "quiet": True,
        "no_warnings": True,
        "default_search": "ytsearch1:",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query_or_url, download=True)
        if "entries" in info and info["entries"]:
            info = info["entries"][0]
    return info.get("title", "YouTube Track"), os.path.join(output_dir, "yt_audio.mp3")


@st.cache_resource(show_spinner=False)
def load_demucs_separator(model_name: str, shifts: int, overlap: float):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return demucs.api.Separator(model=model_name, device=device, shifts=shifts, overlap=overlap)


def trim_audio(audio_path: str, output_dir: str, seconds: int | None):
    if not seconds or seconds <= 0:
        return audio_path
    y, sr = librosa.load(audio_path, sr=44100, duration=seconds, mono=False)
    out = os.path.join(output_dir, f"preview_{seconds}s.wav")
    sf.write(out, y.T if y.ndim > 1 else y, sr)
    return out


def enhance_bass_and_drums(stems: dict, mix_path: str, output_dir: str, sr: int = 44100):
    try:
        mix, _ = librosa.load(mix_path, sr=sr, mono=True)
        if "bass" in stems and os.path.exists(stems["bass"]):
            bass, _ = librosa.load(stems["bass"], sr=sr, mono=True)
            n = min(len(mix), len(bass))
            spec = np.fft.rfft(mix[:n])
            freqs = np.fft.rfftfreq(n, 1 / sr)
            spec[freqs > 180] = 0
            low = np.fft.irfft(spec, n=n)
            enhanced = 0.84 * bass[:n] + 0.16 * low
            enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8) * 0.92
            out = os.path.join(output_dir, "bass_enhanced.wav")
            sf.write(out, enhanced, sr)
            stems["bass"] = out
        if "drums" in stems and os.path.exists(stems["drums"]):
            drums, _ = librosa.load(stems["drums"], sr=sr, mono=True)
            n = min(len(mix), len(drums))
            _, perc = librosa.effects.hpss(mix[:n], margin=(1.0, 3.0))
            enhanced = 0.80 * drums[:n] + 0.20 * perc
            enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8) * 0.92
            out = os.path.join(output_dir, "drums_enhanced.wav")
            sf.write(out, enhanced, sr)
            stems["drums"] = out
    except Exception as exc:
        st.warning(f"Bass/drum post-process atlandı: {exc}")
    return stems


def separate_demucs(audio_path: str, output_dir: str, mode: str, preview_seconds: int | None):
    settings = {
        "Hızlı demo": {"engine": "Demucs", "model": "htdemucs", "shifts": 1, "overlap": 0.10, "enhance": False},
        "Kaliteli pratik": {"engine": "Demucs FT", "model": "htdemucs_ft", "shifts": 2, "overlap": 0.25, "enhance": True},
        "Maksimum kalite": {"engine": "Demucs FT Max", "model": "htdemucs_ft", "shifts": 4, "overlap": 0.50, "enhance": True},
    }[mode]
    work_audio = trim_audio(audio_path, output_dir, preview_seconds)
    sep = load_demucs_separator(settings["model"], settings["shifts"], settings["overlap"])
    _, res = sep.separate_audio_file(work_audio)
    stem_paths = {}
    for name, source in res.items():
        out = os.path.join(output_dir, f"{name}.wav")
        demucs.api.save_audio(source, out, samplerate=sep.samplerate)
        stem_paths[name] = out
    if settings["enhance"]:
        stem_paths = enhance_bass_and_drums(stem_paths, work_audio, output_dir, sep.samplerate)
    return stem_paths, settings, work_audio


def first_matching_file(output_dir: str, terms: list[str]):
    files = list(Path(output_dir).glob("*.wav")) + list(Path(output_dir).glob("*.flac")) + list(Path(output_dir).glob("*.mp3"))
    for p in files:
        low = p.name.lower()
        if any(t in low for t in terms):
            return str(p)
    return None


def separate_uvr(audio_path: str, output_dir: str, preview_seconds: int | None):
    if UVRSeparator is None:
        raise RuntimeError("audio-separator kurulu değil. `uv pip install 'audio-separator[cpu]'` gerekli.")
    work_audio = trim_audio(audio_path, output_dir, preview_seconds)
    settings = {"engine": "UVR/audio-separator", "model": "kuielab A bass/drums + htdemucs_ft fallback", "shifts": "n/a", "overlap": "n/a"}
    stems = {}
    model_plan = {
        "bass": "kuielab_a_bass.onnx",
        "drums": "kuielab_a_drums.onnx",
        "vocals": "kuielab_a_vocals.onnx",
        "other": "kuielab_a_other.onnx",
    }
    for stem_name, model_file in model_plan.items():
        stem_dir = os.path.join(output_dir, f"uvr_{stem_name}")
        os.makedirs(stem_dir, exist_ok=True)
        sep = UVRSeparator(output_dir=stem_dir, output_format="WAV", sample_rate=44100, normalization_threshold=0.92)
        sep.load_model(model_filename=model_file)
        sep.separate(work_audio)
        found = first_matching_file(stem_dir, [stem_name])
        if found:
            out = os.path.join(output_dir, f"{stem_name}_uvr.wav")
            shutil.copyfile(found, out)
            stems[stem_name] = out
    # If any UVR stem is missing, produce a Demucs fallback so the player still works.
    if set(stems) != {"bass", "drums", "vocals", "other"}:
        demucs_stems, _, _ = separate_demucs(audio_path, output_dir, "Kaliteli pratik", preview_seconds)
        demucs_stems.update(stems)
        stems = demucs_stems
    return stems, settings, work_audio


def separate_stems(audio_path: str, output_dir: str, mode: str, preview_seconds: int | None):
    if mode == "UVR deneysel bass/drum":
        return separate_uvr(audio_path, output_dir, preview_seconds)
    return separate_demucs(audio_path, output_dir, mode, preview_seconds)


def make_chord_analysis_audio(stems: dict, fallback_audio: str, output_dir: str, sr: int = 22050):
    """Build a cleaner harmonic source for chord detection.

    Full-mix chord detection is easily confused by drums/vocals/melody.  When
    stems exist, analyze mostly `other` + a little bass, not the whole mix.
    """
    try:
        weights = [("other", 0.85), ("bass", 0.35), ("vocals", 0.10)]
        loaded = []
        for name, weight in weights:
            path = stems.get(name)
            if path and os.path.exists(path):
                y, _ = librosa.load(path, sr=sr, mono=True)
                loaded.append((y, weight))
        if loaded:
            n = min(len(y) for y, _ in loaded)
            mix = np.zeros(n, dtype=np.float32)
            for y, weight in loaded:
                mix += weight * y[:n]
            # Remove percussive residue and normalize softly.
            mix, _ = librosa.effects.hpss(mix)
            mix = mix / (np.max(np.abs(mix)) + 1e-8) * 0.85
            out = os.path.join(output_dir, "chord_analysis_mix.wav")
            sf.write(out, mix, sr)
            return out
    except Exception as exc:
        st.warning(f"Stem tabanlı akor analiz mix'i hazırlanamadı, orijinal mix kullanılacak: {exc}")
    return fallback_audio


CHORD_QUALITIES = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
    "sus4": [0, 5, 7],
    "dim": [0, 3, 6],
}


def chord_templates():
    templates = []
    for root_idx, root in enumerate(NOTES):
        for suffix, intervals in CHORD_QUALITIES.items():
            vec = np.zeros(12)
            for i in intervals:
                vec[(root_idx + i) % 12] = 1.0
            vec[root_idx] = 1.45
            vec[(root_idx + 7) % 12] = max(vec[(root_idx + 7) % 12], 1.05)
            if suffix in ("7", "m7", "maj7"):
                vec[(root_idx + (11 if suffix == "maj7" else 10)) % 12] = 0.82
            vec /= np.linalg.norm(vec) + 1e-8
            templates.append((root, suffix, vec))
    return templates


TEMPLATES = chord_templates()


def normalize_chord_name(chord: str):
    if not chord or chord == "N" or chord == "N.C.":
        return "N.C."
    return chord.replace(":", "").replace("min", "m").replace("maj", "maj")


def chord_root_suffix(chord: str):
    if not chord or chord == "N.C.":
        return None, ""
    for root in sorted(NOTES, key=len, reverse=True):
        if chord.startswith(root):
            return root, chord[len(root):]
    return None, ""


def detect_chord_name(chroma_vec: np.ndarray, key_hint: str | None = None):
    v = np.maximum(chroma_vec, 0).astype(float)
    if float(np.sum(v)) < 1e-7:
        return "N.C.", "-", 0.0
    # Compress outlier melody notes and normalize.
    v = np.sqrt(v)
    v = v / (np.linalg.norm(v) + 1e-8)
    best_score = -1.0
    best = ("N.C.", "-", 0.0)
    key_root = None
    if key_hint:
        key_root, _ = chord_root_suffix(key_hint.replace("m", ""))
    for root, suffix, tmpl in TEMPLATES:
        root_idx = NOTES.index(root)
        score = float(np.dot(v, tmpl))
        # Penalize contradictions that often cause false major/minor flips.
        maj3, min3 = (root_idx + 4) % 12, (root_idx + 3) % 12
        if suffix.startswith("m") and v[maj3] > v[min3] * 1.20:
            score -= 0.10
        if suffix in ("", "7", "maj7") and v[min3] > v[maj3] * 1.20:
            score -= 0.10
        # Mild diatonic/key stability bias; never enough to override clear audio.
        if key_root and root == key_root:
            score += 0.025
        if score > best_score:
            best_score = score
            best = (f"{root}{suffix}", f"{root} ({NOTE_TR[root]})", min(max(score, 0.0), 1.0))
    if best_score < 0.46:
        return "N.C.", "-", best_score
    return best


def estimate_tempo_beats(audio_path: str, duration: int | None):
    y, sr = librosa.load(audio_path, sr=22050, duration=duration, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo_arr = np.asarray(tempo).reshape(-1)
    bpm = int(np.round(float(tempo_arr[0]))) if tempo_arr.size else 120
    if bpm <= 0:
        bpm = 120
    while bpm < 70:
        bpm *= 2
    while bpm > 210:
        bpm = int(round(bpm / 2))
    if len(beats) < 8:
        hop = 512
        step = max(1, int((60 / bpm) * sr / hop))
        beats = np.arange(0, max(1, len(y) // hop), step)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return y, sr, bpm, np.asarray(beats, dtype=int), beat_times


def choose_bar_offset(beats, chroma, y, sr):
    if len(beats) < 8:
        return 0
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    best_offset, best_score = 0, -1e9
    for offset in range(4):
        confs, downbeats = [], []
        for i in range(offset, max(offset, len(beats) - 4), 4):
            if i + 4 >= len(beats):
                break
            seg = chroma[:, beats[i]:beats[i + 4]]
            if seg.size == 0:
                continue
            _, _, conf = detect_chord_name(np.median(seg, axis=1))
            confs.append(conf)
            if beats[i] < len(onset):
                downbeats.append(onset[beats[i]])
        if confs:
            score = float(np.mean(confs)) + 0.04 * float(np.mean(downbeats) if downbeats else 0.0) - 0.015 * offset
            if score > best_score:
                best_score, best_offset = score, offset
    return best_offset


def smooth_bars(bars: list[dict]):
    if len(bars) < 3:
        return bars
    out = [dict(b) for b in bars]
    for i in range(1, len(out) - 1):
        prev_c, cur_c, next_c = out[i - 1]["chord"], out[i]["chord"], out[i + 1]["chord"]
        if prev_c == next_c and cur_c != prev_c and out[i]["confidence"] < 0.78:
            out[i]["chord"] = prev_c
            root, _ = chord_root_suffix(prev_c)
            out[i]["root"] = f"{root} ({NOTE_TR.get(root, root)})" if root else out[i]["root"]
            out[i]["source"] += "+smoothed"
    return out


def bars_from_beat_times(beat_times, bpm: int, audio_len_sec: float | None = None, offset: int = 0):
    bars = []
    if len(beat_times) >= 5:
        for i in range(offset, len(beat_times) - 4, 4):
            start_t = float(beat_times[i])
            end_t = float(beat_times[min(i + 4, len(beat_times) - 1)])
            if end_t > start_t:
                bars.append((start_t, end_t, i, i + 4))
    if not bars:
        bar_len = 4 * 60 / max(bpm, 1)
        total = audio_len_sec or bar_len * 8
        count = max(1, int(np.ceil(total / bar_len)))
        bars = [(i * bar_len, min((i + 1) * bar_len, total), None, None) for i in range(count)]
    return bars


def basic_pitch_chords(audio_path: str, duration: int | None, bpm: int, beat_times, offset: int = 0):
    if basic_pitch_predict is None:
        return None
    try:
        source = trim_audio(audio_path, tempfile.mkdtemp(prefix="basic-pitch-preview-"), duration) if duration else audio_path
        _, _, note_events = basic_pitch_predict(source)
        try:
            audio_len_sec = librosa.get_duration(path=source)
        except Exception:
            audio_len_sec = None
        bar_times = bars_from_beat_times(beat_times, bpm, audio_len_sec, offset=offset)
        bars = []
        for idx, (start_t, end_t, _, _) in enumerate(bar_times, start=1):
            vec = np.zeros(12)
            for ev in note_events:
                n_start, n_end, midi_pitch = float(ev[0]), float(ev[1]), int(ev[2])
                amp = float(ev[3]) if len(ev) > 3 and ev[3] is not None else 1.0
                overlap = max(0.0, min(end_t, n_end) - max(start_t, n_start))
                if overlap <= 0:
                    continue
                pc = midi_pitch % 12
                octave_weight = 1.25 if midi_pitch < 60 else (0.85 if midi_pitch > 76 else 1.0)
                vec[pc] += overlap * max(amp, 0.05) * octave_weight
            chord, root, conf = detect_chord_name(vec)
            if np.sum(vec) <= 1e-6:
                chord, root, conf = "N.C.", "-", 0.0
            bars.append({"bar_num": idx, "chord": chord, "root": root, "confidence": round(float(conf), 2), "start": round(start_t, 3), "end": round(end_t, 3), "source": "basic-pitch-midi"})
        return smooth_bars(bars), note_events
    except Exception as exc:
        st.warning(f"Basic Pitch modeli çalışmadı, fallback kullanılacak: {exc}")
        return None


def infer_key_from_bars(bars: list[dict]):
    counts = {}
    for b in bars:
        c = b.get("chord", "N.C.")
        root, suffix = chord_root_suffix(c)
        if root:
            minor = suffix.startswith("m") and not suffix.startswith("maj")
            k = root + ("m" if minor else "")
            counts[k] = counts.get(k, 0.0) + 1.0 + float(b.get("confidence", 0))
    return max(counts, key=counts.get) if counts else "C"


def detect_chords_for_audio(audio_path: str, duration: int | None = 120, engine: str = "Harmonic Consensus"):
    y, sr, bpm, beats, beat_times = estimate_tempo_beats(audio_path, duration)
    y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 3.0))
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=36)
    chroma_cens = librosa.feature.chroma_cens(y=y_harm, sr=sr)
    chroma = (0.82 * chroma_cqt) + (0.18 * chroma_cens)
    offset = choose_bar_offset(beats, chroma, y, sr)

    if engine.startswith("Basic Pitch"):
        bp = basic_pitch_chords(audio_path, duration, bpm, beat_times, offset=offset)
        if bp is not None:
            bars, _ = bp
            return bpm, infer_key_from_bars(bars), bars

    try:
        audio_len_sec = librosa.get_duration(y=y, sr=sr)
    except Exception:
        audio_len_sec = None
    bar_times = bars_from_beat_times(beat_times, bpm, audio_len_sec, offset=offset)
    essentia_data = essentia_chords(audio_path, duration, bpm) if engine.startswith("Essentia") else None
    bars = []
    key_hint = None
    for idx, (start_t, end_t, start_beat_idx, end_beat_idx) in enumerate(bar_times, start=1):
        if start_beat_idx is not None and end_beat_idx is not None and end_beat_idx < len(beats):
            start_frame, end_frame = beats[start_beat_idx], beats[end_beat_idx]
        else:
            start_frame = librosa.time_to_frames(start_t, sr=sr)
            end_frame = librosa.time_to_frames(end_t, sr=sr)
        if end_frame <= start_frame:
            continue
        vec = np.median(chroma[:, start_frame:end_frame], axis=1)
        chord, root, conf = detect_chord_name(vec, key_hint=key_hint)
        source = f"harmonic-consensus/o{offset}"
        if essentia_data:
            idxs = [j for j, t in enumerate(essentia_data["times"]) if start_t <= float(t) < end_t]
            if idxs:
                candidates = {}
                for j in idxs:
                    c = essentia_data["chords"][j]
                    if c != "N.C.":
                        candidates[c] = candidates.get(c, 0) + essentia_data["strengths"][j]
                if candidates:
                    c_e = max(candidates, key=candidates.get)
                    score_e = min(1.0, candidates[c_e] / max(1, len(idxs)))
                    if score_e > conf + 0.10:
                        chord = c_e
                        root_note, _ = chord_root_suffix(c_e)
                        root = f"{root_note} ({NOTE_TR.get(root_note, root_note)})" if root_note else root
                        conf = score_e
                        source = "essentia-consensus"
        bars.append({"bar_num": len(bars) + 1, "chord": chord, "root": root, "confidence": round(float(conf), 2), "start": round(start_t, 3), "end": round(end_t, 3), "source": source})
        if idx <= 4 and chord != "N.C.":
            key_hint = infer_key_from_bars(bars)
    if not bars:
        bar_len = 4 * 60 / bpm
        bars = [{"bar_num": i + 1, "chord": "N.C.", "root": "-", "confidence": 0.0, "start": round(i * bar_len, 3), "end": round((i + 1) * bar_len, 3), "source": "fallback"} for i in range(8)]
    bars = smooth_bars(bars)
    return bpm, infer_key_from_bars(bars), bars


def make_abc(title: str, bpm: int, key: str, bars: list[dict]):
    key_clean = FLAT_TO_SHARP.get(key, key).replace("maj7", "").replace("m7", "m").replace("7", "")
    if key_clean not in NOTES and not (len(key_clean) > 1 and key_clean[:-1] in NOTES and key_clean.endswith("m")):
        key_clean = "C"
    lines = ["X:1", f"T:{title}", "M:4/4", "L:1/4", f"Q:1/4={bpm}", f"K:{key_clean}"]
    current = []
    for i, bar in enumerate(bars, start=1):
        chord = bar["chord"].replace("#", "♯")
        current.append(f'"{chord}" z4')
        if i % 4 == 0:
            lines.append("| " + " | ".join(current) + " |")
            current = []
    if current:
        lines.append("| " + " | ".join(current) + " |")
    return "\n".join(lines)


def render_practice_workspace(mix_path: str, stems: dict, title: str, abc: str, bars: list[dict], bpm: int):
    track_meta = {"bass": ["Bass", "🎸", "#7c3aed"], "drums": ["Drums", "🥁", "#0284c7"], "vocals": ["Vocals", "🎤", "#db2777"], "other": ["Other", "🎹", "#059669"]}
    tracks = []
    for key in ["bass", "drums", "vocals", "other"]:
        path = stems.get(key)
        if path and os.path.exists(path):
            tracks.append({"id": key, "name": track_meta[key][0], "icon": track_meta[key][1], "color": track_meta[key][2], "src": f"data:{audio_mime(path)};base64,{file_to_b64(path)}"})
    payload = {
        "title": title,
        "mix": f"data:{audio_mime(mix_path)};base64,{file_to_b64(mix_path)}",
        "tracks": tracks,
        "bars": bars,
        "bpm": bpm,
    }
    html_doc = """
<!doctype html><html><head><meta charset="utf-8">
<script type="module">
import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js';
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js';
window.WaveSurfer = WaveSurfer; window.RegionsPlugin = RegionsPlugin;
</script>
<style>
*{box-sizing:border-box} body{margin:0;background:#f6f1e6;color:#171717;font-family:Arial,Helvetica,sans-serif}.wrap{border:1px solid #d8cdb9;background:#f6f1e6;border-radius:12px;overflow:hidden}.player{position:sticky;top:0;z-index:5;background:#f8f4ea;border-bottom:1px solid #d8cdb9;padding:8px 10px}.transport{display:grid;grid-template-columns:auto auto auto auto auto 1fr auto auto;gap:6px;align-items:center}.play{width:34px;height:34px;border-radius:50%;border:1px solid #111;background:#111;color:#fff;font-weight:900;cursor:pointer}.btn{height:28px;border:1px solid #111;border-radius:6px;background:#fffaf0;color:#111;font-weight:800;font-size:11px;cursor:pointer}.btn.active{background:#111;color:#fff}.title{font-weight:800;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.time{font:800 11px ui-monospace,Menlo,monospace}.speed{height:28px;border:1px solid #111;background:#fffaf0;border-radius:6px;font-weight:700}.master{width:82px;accent-color:#111}.wavebox{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-top:6px}.wave{height:34px;border:1px solid #111;background:#fffaf0;border-radius:6px;overflow:hidden}.loop-read{font:800 11px ui-monospace,Menlo,monospace;min-width:125px}.hint{font-size:11px;color:#5f5748;margin-top:4px}.mixer{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:6px}.track{display:grid;grid-template-columns:auto 1fr auto auto 76px;gap:4px;align-items:center;border:1px solid #d8cdb9;background:#fffaf0;border-radius:7px;padding:4px;font-size:11px}.tog{height:21px;min-width:23px;border:1px solid #777;background:white;border-radius:4px;font-weight:900;font-size:10px}.tog.m.active{background:#ef4444;color:white}.tog.s.active{background:#facc15;color:#111}.vol{width:76px;accent-color:#111}.score-panel{padding:12px}.score-title{display:flex;justify-content:space-between;gap:12px;align-items:end;margin-bottom:9px}.song{font-size:22px;font-weight:900;letter-spacing:-.02em}.meta{font:800 12px ui-monospace,Menlo,monospace}.sheet{height:500px;overflow:auto;background:#fffdf7;border:1px solid #111;border-radius:8px;padding:12px}.system{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:0;margin-bottom:18px;border-left:2px solid #111}.measure{position:relative;height:118px;border-right:2px solid #111;background:transparent;cursor:pointer}.measure.active{background:rgba(252,211,77,.25)}.measure.active:after{content:"";position:absolute;top:18px;bottom:14px;left:50%;border-left:3px solid #ef4444}.chord{position:absolute;top:0;left:10px;font-size:23px;font-weight:900}.barno{position:absolute;top:2px;right:7px;font:800 10px ui-monospace,Menlo,monospace;color:#777}.staff{position:absolute;left:0;right:0;top:38px;height:48px}.staff i{position:absolute;left:0;right:0;border-top:1.5px solid #111}.slash{position:absolute;top:50px;left:45%;font-size:28px;font-weight:900;transform:rotate(-18deg)}.root{position:absolute;left:10px;bottom:8px;font-size:10px;color:#555}.section{font:900 12px ui-monospace,Menlo,monospace;margin:10px 0 5px}.empty{font-weight:800;color:#6b6254}.footer-note{font-size:11px;color:#5f5748;margin-top:7px}@media(max-width:760px){.transport{grid-template-columns:auto auto auto auto}.mixer{grid-template-columns:1fr 1fr}.system{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap">
<div class="player"><div class="transport"><button id="play" class="play">▶</button><button class="btn" onclick="jump(-10)">-10</button><button class="btn" onclick="jump(10)">+10</button><button class="btn" onclick="markA()">A koy</button><button class="btn" onclick="markB()">B koy</button><div class="title" id="songTitle"></div><select id="speed" class="speed"><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option></select><input id="master" class="master" title="Master" type="range" min="0" max="1" step="0.01" value="1"><div class="time"><span id="cur">00:00</span>/<span id="dur">00:00</span></div></div>
<div class="wavebox"><div id="wave" class="wave"></div><div class="loop-read" id="loopRead">Loop: kapalı</div></div><div class="hint">Loop: çalarken “A koy”, sonra “B koy”. İstersen waveform üzerinde sürükleyerek de bölge seçebilirsin. Player artık notation scroll’dan ayrı/sabit.</div><div id="mixer" class="mixer"></div></div>
<div class="score-panel"><div class="score-title"><div><div class="song" id="scoreTitle"></div><div class="meta">Real Book staff view • 4 ölçü/satır • slash notation</div></div><div class="meta" id="barStatus">bar -</div></div><div id="sheet" class="sheet"></div><div class="footer-note">Not: Melodi transkripsiyonu henüz basitleştirilmiş slash notation. Akorlar Basic Pitch MIDI notalarından çıkarılıyor; ölçüye tıklayınca player o ölçüye gider.</div></div>
</div><script type="module">
const DATA=__DATA__; const $=id=>document.getElementById(id); const fmt=t=>{if(!isFinite(t))return'00:00';let m=Math.floor(t/60),s=Math.floor(t%60);return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')};
function waitForLibs(){return new Promise(r=>{const t=setInterval(()=>{if(window.WaveSurfer&&window.RegionsPlugin){clearInterval(t);r()}},30)})} await waitForLibs(); $('songTitle').textContent=DATA.title; $('scoreTitle').textContent=DATA.title;
const regions=window.RegionsPlugin.create(); const ws=window.WaveSurfer.create({container:'#wave',url:DATA.mix,waveColor:'#b8aa91',progressColor:'#111',cursorColor:'#ef4444',cursorWidth:2,height:32,barWidth:2,barGap:2,barRadius:0,normalize:true,plugins:[regions]});
let loopA=null, loopB=null, activeRegion=null, loopOn=false, lastSync=0, lastBar=null; regions.enableDragSelection({color:'rgba(17,17,17,.18)'}); regions.on('region-created', r=>{if(activeRegion&&activeRegion.id!==r.id)activeRegion.remove(); activeRegion=r; loopA=r.start; loopB=r.end; loopOn=true; updateLoopRead();});
ws.on('ready',()=>{$('dur').textContent=fmt(ws.getDuration()); if(DATA.tracks.length>0)ws.setVolume(0); else ws.setVolume(parseFloat($('master').value));});
ws.on('timeupdate',t=>{$('cur').textContent=fmt(t); const now=performance.now(); if(now-lastSync>500){syncStemTime(t); lastSync=now;} updateBar(t); if(loopOn&&loopA!=null&&loopB!=null&&t>=loopB)seekAll(loopA,true);});
ws.on('play',()=>{const t=ws.getCurrentTime(); Object.values(tracks).forEach(tr=>{tr.audio.currentTime=t; tr.audio.play()}); $('play').textContent='Ⅱ'}); ws.on('pause',()=>{Object.values(tracks).forEach(t=>t.audio.pause()); $('play').textContent='▶'}); $('play').onclick=()=>ws.playPause();
window.markA=()=>{loopA=ws.getCurrentTime(); if(loopB!=null&&loopB<=loopA)loopB=null; redrawRegion(); updateLoopRead();}; window.markB=()=>{loopB=ws.getCurrentTime(); if(loopA==null||loopB<=loopA){loopA=Math.max(0,loopB-4);} loopOn=true; redrawRegion(); updateLoopRead();}; function redrawRegion(){if(activeRegion)activeRegion.remove(); if(loopA!=null&&loopB!=null)activeRegion=regions.addRegion({start:loopA,end:loopB,color:'rgba(17,17,17,.18)',drag:true,resize:true});} function updateLoopRead(){$('loopRead').textContent=(loopA!=null&&loopB!=null)?`Loop: ${fmt(loopA)}-${fmt(loopB)}`:'Loop: kapalı'}
const tracks={}; function buildMixer(){const m=$('mixer'); if(DATA.tracks.length===0){m.innerHTML='<div class="empty">Stem yok: orijinal mix çalıyor.</div>';return;} DATA.tracks.forEach(tr=>{const a=new Audio(tr.src); a.preload='auto'; tracks[tr.id]={...tr,audio:a,mute:false,solo:false,vol:1}; const row=document.createElement('div'); row.className='track'; row.innerHTML=`<b style="color:${tr.color}">${tr.icon}</b><span>${tr.name}</span><button id="m_${tr.id}" class="tog m">M</button><button id="s_${tr.id}" class="tog s">S</button><input id="v_${tr.id}" class="vol" type="range" min="0" max="1.5" step="0.01" value="1">`; m.appendChild(row); $(`m_${tr.id}`).onclick=()=>{tracks[tr.id].mute=!tracks[tr.id].mute;$(`m_${tr.id}`).classList.toggle('active',tracks[tr.id].mute);applyMix()}; $(`s_${tr.id}`).onclick=()=>{tracks[tr.id].solo=!tracks[tr.id].solo;$(`s_${tr.id}`).classList.toggle('active',tracks[tr.id].solo);applyMix()}; $(`v_${tr.id}`).oninput=e=>{tracks[tr.id].vol=parseFloat(e.target.value);applyMix()};});}
function applyMix(){const anySolo=Object.values(tracks).some(t=>t.solo); const master=parseFloat($('master').value); if(DATA.tracks.length===0)ws.setVolume(master); Object.values(tracks).forEach(t=>{let v=t.vol*master; if(t.mute||(anySolo&&!t.solo))v=0; t.audio.volume=Math.max(0,Math.min(1,v));});} $('master').oninput=applyMix; $('speed').onchange=e=>{let r=parseFloat(e.target.value); ws.setPlaybackRate(r,true); Object.values(tracks).forEach(t=>t.audio.playbackRate=r);};
function seekAll(time,play=false){ws.setTime(time); Object.values(tracks).forEach(t=>{t.audio.currentTime=time;if(play)t.audio.play();});} function syncStemTime(time){Object.values(tracks).forEach(t=>{if(Math.abs(t.audio.currentTime-time)>.22)t.audio.currentTime=time;});} window.jump=s=>seekAll(Math.max(0,Math.min(ws.getDuration(),ws.getCurrentTime()+s)),ws.isPlaying()); buildMixer(); applyMix();
function renderSheet(){const sheet=$('sheet'); if(!DATA.bars.length){sheet.innerHTML='<div class="empty">Notation üretilemedi.</div>';return;} for(let i=0;i<DATA.bars.length;i+=4){const sec=document.createElement('div'); sec.className='section'; sec.textContent=(i===0?'[A]':(i%16===0?'[B]':'')); if(sec.textContent)sheet.appendChild(sec); const sys=document.createElement('div'); sys.className='system'; DATA.bars.slice(i,i+4).forEach(b=>{const d=document.createElement('div'); d.className='measure'; d.id='bar_'+b.bar_num; d.onclick=()=>seekAll(b.start,ws.isPlaying()); d.innerHTML=`<div class="chord">${String(b.chord).replaceAll('#','♯')}</div><div class="barno">${b.bar_num}</div><div class="staff"><i style="top:0"></i><i style="top:12px"></i><i style="top:24px"></i><i style="top:36px"></i><i style="top:48px"></i></div><div class="slash">/</div><div class="root">${b.root||''} · ${Math.round((b.confidence||0)*100)}%</div>`; sys.appendChild(d);}); sheet.appendChild(sys);}}
function updateBar(t){let b=DATA.bars.find(x=>t>=x.start&&t<x.end)||DATA.bars[DATA.bars.length-1]; if(!b||b.bar_num===lastBar)return; if(lastBar)$('bar_'+lastBar)?.classList.remove('active'); lastBar=b.bar_num; const el=$('bar_'+lastBar); el?.classList.add('active'); el?.scrollIntoView({block:'nearest',inline:'nearest'}); $('barStatus').textContent=`bar ${b.bar_num} • ${b.chord}`;} renderSheet();
</script></body></html>
""".replace("__DATA__", json.dumps(payload))
    components.html(html_doc, height=760, scrolling=False)


st.markdown("<div class='app-hero'><div class='brand'>🎸 <span>Bass Studio</span></div><div class='pill'>minimal player • Real Book staff • local Mac tools</div></div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([3.6, 1.7, 1.25, 1.6, 1.0])
    with c1:
        query = st.text_input("YouTube", placeholder="YouTube linki veya şarkı adı...", label_visibility="collapsed")
    with c2:
        mode = st.selectbox("Stem", ["Kaliteli pratik", "Hızlı demo", "Maksimum kalite", "UVR deneysel bass/drum", "Sadece akor/notasyon"], label_visibility="collapsed")
    with c3:
        scope = st.selectbox("Süre", ["İlk 90 sn", "İlk 180 sn", "Tam şarkı"], label_visibility="collapsed")
    with c4:
        chord_engine = st.selectbox("Akor", ["Harmonic Consensus", "Essentia Consensus", "Basic Pitch MIDI + Music21", "Librosa ensemble"], label_visibility="collapsed")
    with c5:
        run = st.button("Yükle", type="primary", use_container_width=True)
    st.markdown("<div class='small-note'>Öneri: önce Kaliteli pratik + İlk 90 sn. UVR deneysel mod ilk kullanımda model indirebilir ve uzun sürebilir.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if run and query:
    temp_dir = tempfile.mkdtemp(prefix="bass-studio-")
    with st.spinner("YouTube ses kaynağı hazırlanıyor..."):
        title, audio_path = download_youtube_audio(query, temp_dir)
    st.session_state.current_title = title
    st.session_state.current_path = audio_path
    st.session_state.temp_dir = temp_dir
    st.session_state.process_key = None

if "current_path" in st.session_state and os.path.exists(st.session_state.current_path):
    title = st.session_state.current_title
    audio_path = st.session_state.current_path
    temp_dir = st.session_state.temp_dir
    preview_seconds = {"İlk 90 sn": 90, "İlk 180 sn": 180, "Tam şarkı": None}[scope]
    process_key = f"{title}|{mode}|{scope}|{chord_engine}"
    st.subheader(title)
    if st.session_state.get("process_key") != process_key:
        with st.spinner("Analiz hazırlanıyor..."):
            work_audio = trim_audio(audio_path, temp_dir, preview_seconds)
            if mode == "Sadece akor/notasyon":
                stems, sep_settings = {}, {"engine": "none", "model": "chord-only", "shifts": "-", "overlap": "-"}
            else:
                stems, sep_settings, work_audio = separate_stems(audio_path, temp_dir, mode, preview_seconds)
            analysis_audio = make_chord_analysis_audio(stems, work_audio, temp_dir)
            bpm, key, bars = detect_chords_for_audio(analysis_audio, duration=None, engine=chord_engine)
            st.session_state.analysis_audio = analysis_audio
            st.session_state.stems = stems
            st.session_state.work_audio = work_audio
            st.session_state.sep_settings = sep_settings
            st.session_state.bpm = bpm
            st.session_state.key = key
            st.session_state.bars = bars
            st.session_state.abc = make_abc(title, bpm, key, bars)
            st.session_state.process_key = process_key
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stem engine", st.session_state.sep_settings["engine"])
    m2.metric("Model", str(st.session_state.sep_settings["model"])[:28])
    m3.metric("Tempo", f"{st.session_state.bpm} BPM")
    m4.metric("Ölçü", f"{len(st.session_state.bars)} bars")
    render_practice_workspace(st.session_state.work_audio, st.session_state.stems, title, st.session_state.abc, st.session_state.bars, st.session_state.bpm)
    with st.expander("Teknik çıktı / tespit edilen akorlar"):
        st.code(st.session_state.abc, language="text")
        st.json(st.session_state.bars[:64])
else:
    st.info("Bir şarkı adı veya YouTube linki girip Yükle'ye basın.")
