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


CHORD_QUALITIES = {
    "": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11], "dim": [0, 3, 6], "sus4": [0, 5, 7], "6": [0, 4, 7, 9], "m6": [0, 3, 7, 9]
}


def chord_templates():
    templates = []
    for root_idx, root in enumerate(NOTES):
        for suffix, intervals in CHORD_QUALITIES.items():
            vec = np.zeros(12)
            for i in intervals:
                vec[(root_idx + i) % 12] = 1.0
            vec[root_idx] = 1.28
            vec /= np.linalg.norm(vec) + 1e-8
            templates.append((root, suffix, vec))
    return templates


TEMPLATES = chord_templates()


def normalize_chord_name(chord: str):
    if not chord or chord == "N" or chord == "N.C.":
        return "N.C."
    chord = chord.replace(":", "").replace("min", "m").replace("maj", "maj")
    return chord


def detect_chord_name(chroma_vec: np.ndarray):
    v = np.maximum(chroma_vec, 0)
    v = v / (np.linalg.norm(v) + 1e-8)
    root_energy = v / (np.max(v) + 1e-8)
    best_score = -1
    best = ("N.C.", "-", 0.0)
    for root, suffix, tmpl in TEMPLATES:
        score = float(np.dot(v, tmpl))
        root_bonus = 0.08 * float(root_energy[NOTES.index(root)])
        score += root_bonus
        if score > best_score:
            best_score = score
            best = (f"{root}{suffix}", f"{root} ({NOTE_TR[root]})", min(score, 1.0))
    if best_score < 0.50:
        return "N.C.", "-", best_score
    return best


def essentia_chords(audio_path: str, duration: int | None, bpm_fallback: int):
    if es is None:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=44100, duration=duration, mono=True)
        if not len(y):
            return None
        key, scale, key_strength = es.KeyExtractor(sampleRate=sr)(y.astype("float32"))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
        pcp = chroma.T.astype("float32")
        chords, strengths = es.ChordsDetection(hopSize=512, sampleRate=sr)(pcp)
        times = librosa.frames_to_time(np.arange(len(chords)), sr=sr, hop_length=512)
        return {"chords": [normalize_chord_name(c) for c in chords], "strengths": list(map(float, strengths)), "times": times, "key": f"{key}{'m' if scale == 'minor' else ''}", "key_strength": float(key_strength)}
    except Exception:
        return None


def detect_chords_for_audio(audio_path: str, duration: int | None = 120, engine: str = "Essentia + Librosa ensemble"):
    y, sr = librosa.load(audio_path, sr=22050, duration=duration, mono=True)
    y_harm, _ = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo_arr = np.asarray(tempo).reshape(-1)
    bpm = int(np.round(float(tempo_arr[0]))) if tempo_arr.size else 120
    if bpm <= 0:
        bpm = 120
    if len(beats) < 8:
        step = max(1, int((60 / bpm) * sr / 512))
        beats = librosa.samples_to_frames(librosa.frames_to_samples(librosa.util.fix_frames(np.arange(0, len(y) // 512, step))))
    beat_times = librosa.frames_to_time(beats, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=36)
    chroma_cens = librosa.feature.chroma_cens(y=y_harm, sr=sr)
    essentia_data = essentia_chords(audio_path, duration, bpm) if engine.startswith("Essentia") else None
    bars = []
    for i in range(0, max(0, len(beats) - 4), 4):
        start_frame, end_frame = beats[i], beats[min(i + 4, len(beats) - 1)]
        if end_frame <= start_frame:
            continue
        vec1 = np.median(chroma_cqt[:, start_frame:end_frame], axis=1)
        vec2 = np.median(chroma_cens[:, start_frame:end_frame], axis=1)
        chord_l, root_l, conf_l = detect_chord_name((0.72 * vec1) + (0.28 * vec2))
        start_t = float(beat_times[i]) if i < len(beat_times) else len(bars) * 4 * 60 / bpm
        end_t = float(beat_times[min(i + 4, len(beat_times) - 1)]) if len(beat_times) else start_t + 4 * 60 / bpm
        chord, root, conf, source = chord_l, root_l, conf_l, "librosa-ensemble"
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
                    # Prefer Essentia when it is confident, otherwise keep the more stable template result.
                    if score_e >= 0.45 or conf_l < 0.58:
                        chord = c_e
                        root_note = c_e.replace("maj7", "").replace("m7", "").replace("dim", "").replace("sus4", "").replace("7", "").replace("6", "").replace("m", "")
                        root = f"{root_note} ({NOTE_TR.get(root_note, root_note)})" if root_note in NOTE_TR else root_l
                        conf = max(score_e, conf_l * 0.92)
                        source = "essentia+librosa"
        bars.append({"bar_num": len(bars) + 1, "chord": chord, "root": root, "confidence": round(float(conf), 2), "start": round(start_t, 3), "end": round(end_t, 3), "source": source})
    if not bars:
        bar_len = 4 * 60 / bpm
        bars = [{"bar_num": i + 1, "chord": "N.C.", "root": "-", "confidence": 0.0, "start": round(i * bar_len, 3), "end": round((i + 1) * bar_len, 3), "source": "fallback"} for i in range(8)]
    key = essentia_data["key"] if essentia_data and essentia_data.get("key") else "C"
    if key == "C":
        for b in bars:
            if b["chord"] != "N.C.":
                key = b["chord"]
                for suffix in ("maj7", "m7", "dim", "sus4", "7", "m6", "6", "m"):
                    if key.endswith(suffix):
                        key = key[: -len(suffix)] + ("m" if suffix in ("m", "m7", "m6") else "")
                        break
                break
    return bpm, key, bars


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
    track_meta = {"bass": ["Bass", "🎸", "#a78bfa"], "drums": ["Drums", "🥁", "#38bdf8"], "vocals": ["Vocals", "🎤", "#f472b6"], "other": ["Other", "🎹", "#34d399"]}
    tracks = []
    for key in ["bass", "drums", "vocals", "other"]:
        path = stems.get(key)
        if path and os.path.exists(path):
            tracks.append({"id": key, "name": track_meta[key][0], "icon": track_meta[key][1], "color": track_meta[key][2], "src": f"data:{audio_mime(path)};base64,{file_to_b64(path)}"})
    payload = {
        "title": title,
        "mix": f"data:{audio_mime(mix_path)};base64,{file_to_b64(mix_path)}",
        "tracks": tracks,
        "abc": abc,
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
<script src="https://cdn.jsdelivr.net/npm/abcjs@6.7.0/dist/abcjs-basic-min.js"></script>
<style>
*{box-sizing:border-box} body{margin:0;background:transparent;color:#e5e7eb;font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}.shell{background:linear-gradient(180deg,#101827,#090d17);border:1px solid rgba(255,255,255,.09);border-radius:22px;padding:10px;box-shadow:0 20px 70px rgba(0,0,0,.32)}.top{display:grid;grid-template-columns:auto auto auto 1fr auto auto;align-items:center;gap:8px;margin-bottom:10px}.play{width:44px;height:44px;border-radius:50%;border:0;background:linear-gradient(135deg,#7c3aed,#06b6d4);color:white;font-size:17px;font-weight:900;cursor:pointer}.btn{height:32px;border-radius:10px;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.06);color:#dbeafe;font-weight:850;font-size:12px;cursor:pointer}.btn.active{background:#22d3ee;color:#06111c}.title{font-weight:850;letter-spacing:-.03em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-left:5px}.time{font:800 12px ui-monospace,Menlo,monospace;color:#9ca3af}.speed,.master{height:32px;border-radius:10px;background:#111827;color:#e5e7eb;border:1px solid rgba(255,255,255,.1);font-weight:750}.master{width:96px;accent-color:#22d3ee}.wave-card{position:relative;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:7px 10px;margin-bottom:6px}.hint{font:700 11px ui-monospace,Menlo,monospace;color:#94a3b8;margin-top:5px}.mixer{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:6px 0 8px}.track{display:grid;grid-template-columns:1fr auto auto;gap:6px;align-items:center;padding:8px;border-radius:13px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.065)}.track-name{font-weight:850;font-size:13px;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.tog{height:24px;width:28px;border:0;border-radius:7px;background:#1f2937;color:#9ca3af;font-weight:900;font-size:11px;cursor:pointer}.tog.m.active{background:#ef4444;color:white}.tog.s.active{background:#facc15;color:#111827}.vol{grid-column:1/4;width:100%;accent-color:var(--c);height:4px}.score{position:relative;background:#fffdf6;color:#111827;border-radius:18px;border:1px solid #d7cfbd;padding:18px;min-height:880px;max-height:none;overflow:visible}.score-head{display:flex;justify-content:space-between;align-items:center;font:800 11px ui-monospace,Menlo,monospace;color:#746b5a;margin-bottom:8px}.bargrid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.barcell{position:relative;border:1px solid #ded6c4;border-radius:12px;padding:14px 10px;text-align:center;background:#fffaf0;transition:.12s;min-height:108px;display:flex;flex-direction:column;justify-content:center}.barcell.active{background:#dbeafe;border-color:#2563eb;box-shadow:inset 0 0 0 2px #2563eb}.barcell.active:before{content:"";position:absolute;top:-7px;bottom:-7px;left:50%;width:2px;background:#ef4444}.barno{font:800 12px ui-monospace,Menlo,monospace;color:#8a806c}.ch{font-size:34px;font-weight:950;color:#111827;line-height:1.05}.rt{font-size:12px;color:#6b7280;margin-top:4px}.conf{font-size:10px;color:#9ca3af;margin-top:4px}.long-note{font:800 13px ui-monospace,Menlo,monospace;color:#746b5a;margin-bottom:8px}.presets{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}@media(max-width:820px){.top{grid-template-columns:auto auto auto 1fr}.mixer{grid-template-columns:1fr 1fr}.bargrid{grid-template-columns:repeat(2,1fr)}.master,.speed{width:80px}.title{grid-column:1/5}}
</style></head><body><div class="shell"><div class="top"><button id="play" class="play">▶</button><button class="btn" onclick="jump(-10)">↶10</button><button class="btn" onclick="jump(10)">10↷</button><div class="title" id="songTitle"></div><select id="speed" class="speed"><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option></select><input id="master" class="master" title="Master" type="range" min="0" max="1" step="0.01" value="1"><div class="time"><span id="cur">00:00</span> / <span id="dur">00:00</span></div></div><div class="wave-card"><div id="wave"></div><div class="presets"><button id="loopBtn" class="btn">Loop off</button><button class="btn" onclick="clearLoop()">Loop temizle</button><button class="btn" onclick="preset('groove')">Bass+Drums</button><button class="btn" onclick="preset('backing')">Backing</button><button class="btn" onclick="preset('all')">All</button></div><div class="hint">Loop için waveform üzerinde sürükleyip bölge seç. Bölgeyi tutup taşıyabilir, kenarlardan uzatabilirsin.</div></div><div id="mixer" class="mixer"></div><div class="score"><div class="score-head"><span>abcjs notation + aktif ölçü takibi</span><span id="barStatus">bar -</span></div><div id="score"></div><div id="bargrid" class="bargrid"></div></div></div>
<script type="module">
const DATA = __DATA__;
function waitForLibs(){return new Promise(r=>{const t=setInterval(()=>{if(window.WaveSurfer&&window.RegionsPlugin&&window.ABCJS){clearInterval(t);r()}},30)})}
await waitForLibs();
const $=id=>document.getElementById(id); $('songTitle').textContent=DATA.title;
const fmt=t=>{if(!isFinite(t))return'00:00';let m=Math.floor(t/60),s=Math.floor(t%60);return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')};
const regions = window.RegionsPlugin.create();
const ws = window.WaveSurfer.create({container:'#wave', url:DATA.mix, waveColor:'#334155', progressColor:'#22d3ee', cursorColor:'#ef4444', cursorWidth:2, height:54, barWidth:2, barGap:1, barRadius:2, normalize:true, plugins:[regions]});
regions.enableDragSelection({color:'rgba(168,85,247,.25)'}); let activeRegion=null, loopOn=false;
regions.on('region-created', r=>{ if(activeRegion && activeRegion.id!==r.id) activeRegion.remove(); activeRegion=r; loopOn=true; $('loopBtn').classList.add('active'); $('loopBtn').textContent='Loop on'; });
regions.on('region-clicked', (r,e)=>{ e.stopPropagation(); activeRegion=r; r.play(); });
ws.on('ready', d=>{$('dur').textContent=fmt(ws.getDuration()); if(DATA.tracks.length>0) ws.setVolume(0); else ws.setVolume(parseFloat($('master').value));});
let lastSync=0;
ws.on('timeupdate', t=>{ $('cur').textContent=fmt(t); const now=performance.now(); if(now-lastSync>420){ syncStemTime(t); lastSync=now; } updateBar(t); if(loopOn&&activeRegion&&t>=activeRegion.end) seekAll(activeRegion.start,true); });
ws.on('play',()=>{const t=ws.getCurrentTime(); Object.values(tracks).forEach(tr=>{tr.audio.currentTime=t; tr.audio.play()}); $('play').textContent='Ⅱ'}); ws.on('pause',()=>{Object.values(tracks).forEach(t=>t.audio.pause()); $('play').textContent='▶'});
$('play').onclick=()=>ws.playPause(); $('loopBtn').onclick=()=>{loopOn=!loopOn; $('loopBtn').classList.toggle('active',loopOn); $('loopBtn').textContent=loopOn?'Loop on':'Loop off'}; window.clearLoop=()=>{if(activeRegion)activeRegion.remove(); activeRegion=null; loopOn=false; $('loopBtn').classList.remove('active'); $('loopBtn').textContent='Loop off'};
const tracks={}; const mix=document.createElement('audio');
function buildMixer(){const m=$('mixer'); DATA.tracks.forEach(tr=>{const a=new Audio(tr.src); a.preload='auto'; a.volume=1; tracks[tr.id]={...tr,audio:a,mute:false,solo:false,vol:1}; const row=document.createElement('div'); row.className='track'; row.style.setProperty('--c',tr.color); row.innerHTML=`<div class="track-name" style="color:${tr.color}">${tr.icon} ${tr.name}</div><button id="m_${tr.id}" class="tog m">M</button><button id="s_${tr.id}" class="tog s">S</button><input id="v_${tr.id}" class="vol" type="range" min="0" max="1.5" step="0.01" value="1">`; m.appendChild(row); $(`m_${tr.id}`).onclick=()=>{tracks[tr.id].mute=!tracks[tr.id].mute; $(`m_${tr.id}`).classList.toggle('active',tracks[tr.id].mute); applyMix()}; $(`s_${tr.id}`).onclick=()=>{tracks[tr.id].solo=!tracks[tr.id].solo; $(`s_${tr.id}`).classList.toggle('active',tracks[tr.id].solo); applyMix()}; $(`v_${tr.id}`).oninput=e=>{tracks[tr.id].vol=parseFloat(e.target.value); applyMix()}; });}
function applyMix(){const anySolo=Object.values(tracks).some(t=>t.solo); const master=parseFloat($('master').value); if(DATA.tracks.length===0) ws.setVolume(master); Object.values(tracks).forEach(t=>{let v=t.vol*master; if(t.mute||(anySolo&&!t.solo))v=0; t.audio.volume=Math.max(0,Math.min(1,v));});}
$('master').oninput=applyMix; $('speed').onchange=e=>{ws.setPlaybackRate(parseFloat(e.target.value), true); Object.values(tracks).forEach(t=>t.audio.playbackRate=parseFloat(e.target.value));};
function seekAll(time, play=false){ws.setTime(time); Object.values(tracks).forEach(t=>{t.audio.currentTime=time; if(play)t.audio.play();});}
function syncStemTime(time){Object.values(tracks).forEach(t=>{if(Math.abs(t.audio.currentTime-time)>.18)t.audio.currentTime=time;});}
window.jump=s=>seekAll(Math.max(0, Math.min(ws.getDuration(), ws.getCurrentTime()+s)), ws.isPlaying());
window.preset=name=>{Object.keys(tracks).forEach(id=>{tracks[id].mute=false;tracks[id].solo=false;$(`m_${id}`).classList.remove('active');$(`s_${id}`).classList.remove('active')}); if(name==='groove'){['bass','drums'].forEach(id=>{if(tracks[id]){tracks[id].solo=true;$(`s_${id}`).classList.add('active')}})} if(name==='backing'&&tracks.bass){tracks.bass.mute=true;$('m_bass').classList.add('active')} applyMix();};
buildMixer(); applyMix();
if (DATA.bars.length <= 32) { ABCJS.renderAbc('score', DATA.abc, {responsive:'resize', staffwidth:980, paddingtop:0, paddingbottom:0}); } else { $('score').innerHTML = '<div class=\"long-note\">Uzun parçada performans için staff çizimi kapalı; tam sayfa chord sheet aktif.</div>'; }
const grid=$('bargrid'); DATA.bars.forEach(b=>{const d=document.createElement('div'); d.className='barcell'; d.id='bar_'+b.bar_num; d.onclick=()=>seekAll(b.start, ws.isPlaying()); d.innerHTML=`<div class="barno">${b.bar_num}</div><div class="ch">${String(b.chord).replaceAll('#','♯')}</div><div class="rt">${b.root}</div><div class="conf">${b.source||''} ${Math.round((b.confidence||0)*100)}%</div>`; grid.appendChild(d)});
let lastBar=null; function updateBar(t){let b=DATA.bars.find(x=>t>=x.start&&t<x.end)||DATA.bars[DATA.bars.length-1]; if(!b||b.bar_num===lastBar)return; if(lastBar)$('bar_'+lastBar)?.classList.remove('active'); lastBar=b.bar_num; $('bar_'+lastBar)?.classList.add('active'); $('bar_'+lastBar)?.scrollIntoView({block:'center',inline:'nearest',behavior:'smooth'}); $('barStatus').textContent=`bar ${b.bar_num} • ${b.chord}`;}
</script></body></html>
""".replace("__DATA__", json.dumps(payload))
    components.html(html_doc, height=1280, scrolling=True)


st.markdown("<div class='app-hero'><div class='brand'>🎸 <span>Bass Studio</span></div><div class='pill'>wavesurfer loop • compact mixer • synced notation</div></div>", unsafe_allow_html=True)

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
        chord_engine = st.selectbox("Akor", ["Essentia + Librosa ensemble", "Librosa ensemble"], label_visibility="collapsed")
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
            bpm, key, bars = detect_chords_for_audio(work_audio, duration=None, engine=chord_engine)
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
        st.code(st.session_state.abc, language="abc")
        st.json(st.session_state.bars[:64])
else:
    st.info("Bir şarkı adı veya YouTube linki girip Yükle'ye basın.")
