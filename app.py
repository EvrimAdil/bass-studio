import base64
import json
import os
import tempfile

import demucs.api
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import torch
import yt_dlp

st.set_page_config(
    page_title="Bass Studio",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
NOTE_TR = {
    "C": "Do",
    "C#": "Do#",
    "D": "Re",
    "D#": "Re#",
    "E": "Mi",
    "F": "Fa",
    "F#": "Fa#",
    "G": "Sol",
    "G#": "Sol#",
    "A": "La",
    "A#": "La#",
    "B": "Si",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650;800&family=IBM+Plex+Mono:wght@500;700&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { background: #070a12; color: #eef2ff; font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
.main .block-container { max-width: 1180px; padding: 1.1rem 1.4rem 3rem; }
#MainMenu, header, footer { visibility: hidden; }
.app-hero { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:12px 0 18px; border-bottom:1px solid rgba(255,255,255,.075); margin-bottom:18px; }
.brand { font-size:1.62rem; font-weight:800; letter-spacing:-.04em; }
.brand span { background:linear-gradient(120deg,#8b5cf6,#22d3ee 55%,#34d399); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.pill { color:#a7b0c8; border:1px solid rgba(255,255,255,.1); border-radius:999px; padding:7px 11px; font:700 .75rem 'IBM Plex Mono', monospace; background:rgba(255,255,255,.035); }
.card { border:1px solid rgba(255,255,255,.075); background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.025)); border-radius:18px; padding:16px; box-shadow:0 20px 50px rgba(0,0,0,.25); }
.small-note { color:#94a3b8; font-size:.85rem; }
.stButton > button { border-radius:14px !important; font-weight:750 !important; }
.stTextInput input, .stSelectbox div[data-baseweb="select"] { border-radius:14px !important; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:80]


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def download_youtube_audio(query_or_url: str, output_dir: str):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "yt_audio.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "256"}
        ],
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
def load_separator(model_name: str, shifts: int, overlap: float):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return demucs.api.Separator(model=model_name, device=device, shifts=shifts, overlap=overlap)


def trim_audio(audio_path: str, output_dir: str, seconds: int | None):
    if not seconds or seconds <= 0:
        return audio_path
    y, sr = librosa.load(audio_path, sr=44100, duration=seconds, mono=False)
    out = os.path.join(output_dir, f"preview_{seconds}s.wav")
    if y.ndim == 1:
        sf.write(out, y, sr)
    else:
        sf.write(out, y.T, sr)
    return out


def enhance_bass_and_drums(stems: dict, mix_path: str, output_dir: str, sr: int = 44100):
    """Light post-processing to make bass/drums more practice-friendly.
    Bass: reinforces low-end below ~180Hz from the original mix.
    Drums: reinforces percussive transients via HPSS percussive component.
    This does not replace Demucs; it cleans the practice stems after separation.
    """
    try:
        mix, _ = librosa.load(mix_path, sr=sr, mono=True)
        if "bass" in stems and os.path.exists(stems["bass"]):
            bass, _ = librosa.load(stems["bass"], sr=sr, mono=True)
            n = min(len(mix), len(bass))
            mix_low = librosa.effects.preemphasis(mix[:n], coef=-0.97)
            # Low-pass by zeroing high frequency bins.
            spec = np.fft.rfft(mix_low)
            freqs = np.fft.rfftfreq(n, 1 / sr)
            spec[freqs > 180] = 0
            reinforced_low = np.fft.irfft(spec, n=n)
            enhanced = 0.82 * bass[:n] + 0.18 * reinforced_low
            enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8) * 0.92
            out = os.path.join(output_dir, "bass_enhanced.wav")
            sf.write(out, enhanced, sr)
            stems["bass"] = out

        if "drums" in stems and os.path.exists(stems["drums"]):
            drums, _ = librosa.load(stems["drums"], sr=sr, mono=True)
            n = min(len(mix), len(drums))
            _, perc = librosa.effects.hpss(mix[:n], margin=(1.0, 3.0))
            enhanced = 0.78 * drums[:n] + 0.22 * perc
            enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8) * 0.92
            out = os.path.join(output_dir, "drums_enhanced.wav")
            sf.write(out, enhanced, sr)
            stems["drums"] = out
    except Exception as exc:
        st.warning(f"Bass/drum post-process atlandı: {exc}")
    return stems


def separate_stems(audio_path: str, output_dir: str, mode: str, preview_seconds: int | None):
    settings = {
        "Hızlı demo": {"model": "htdemucs", "shifts": 1, "overlap": 0.10, "enhance": False},
        "Kaliteli pratik": {"model": "htdemucs_ft", "shifts": 2, "overlap": 0.25, "enhance": True},
        "Maksimum kalite": {"model": "htdemucs_ft", "shifts": 4, "overlap": 0.50, "enhance": True},
    }[mode]
    work_audio = trim_audio(audio_path, output_dir, preview_seconds)
    sep = load_separator(settings["model"], settings["shifts"], settings["overlap"])
    _, res = sep.separate_audio_file(work_audio)
    stem_paths = {}
    for name, source in res.items():
        out = os.path.join(output_dir, f"{name}.wav")
        demucs.api.save_audio(source, out, samplerate=sep.samplerate)
        stem_paths[name] = out
    if settings["enhance"]:
        stem_paths = enhance_bass_and_drums(stem_paths, work_audio, output_dir, sep.samplerate)
    return stem_paths, settings


def chord_templates():
    qualities = {
        "": [0, 4, 7],
        "m": [0, 3, 7],
        "7": [0, 4, 7, 10],
        "m7": [0, 3, 7, 10],
        "maj7": [0, 4, 7, 11],
        "dim": [0, 3, 6],
        "sus4": [0, 5, 7],
    }
    templates = []
    for root_idx, root in enumerate(NOTES):
        for suffix, intervals in qualities.items():
            vec = np.zeros(12)
            for i in intervals:
                vec[(root_idx + i) % 12] = 1.0
            vec[root_idx] = 1.25
            vec /= np.linalg.norm(vec) + 1e-8
            templates.append((root, suffix, vec))
    return templates


TEMPLATES = chord_templates()


def detect_chord_name(chroma_vec: np.ndarray):
    v = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-8)
    best = max(TEMPLATES, key=lambda item: float(np.dot(v, item[2])))
    root, suffix, score = best[0], best[1], float(np.dot(v, best[2]))
    if score < 0.54:
        return "N.C.", "-", score
    return f"{root}{suffix}", f"{root} ({NOTE_TR[root]})", score


def detect_chords_for_audio(audio_path: str, duration: int | None = 120):
    y, sr = librosa.load(audio_path, sr=22050, duration=duration, mono=True)
    y_harm, _ = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo_arr = np.asarray(tempo).reshape(-1)
    bpm = int(np.round(float(tempo_arr[0]))) if tempo_arr.size else 120
    if bpm <= 0:
        bpm = 120
    if len(beats) < 8:
        beats = librosa.frames_to_samples(librosa.util.fix_frames(np.arange(0, len(y) // 512, max(1, int((60 / bpm) * sr / 512)))))
        beats = librosa.samples_to_frames(beats)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=36)
    bars = []
    for i in range(0, max(0, len(beats) - 4), 4):
        start, end = beats[i], beats[min(i + 4, len(beats) - 1)]
        if end <= start:
            continue
        vec = np.median(chroma[:, start:end], axis=1)
        chord, root, conf = detect_chord_name(vec)
        if bars and bars[-1]["chord"] == chord:
            # keep repeated bars visible but confidence smoothed
            conf = (bars[-1].get("confidence", conf) + conf) / 2
        bars.append({"bar_num": len(bars) + 1, "chord": chord, "root": root, "confidence": round(conf, 2)})
    if not bars:
        bars = [{"bar_num": i + 1, "chord": "N.C.", "root": "-", "confidence": 0.0} for i in range(8)]
    key = "C"
    for b in bars:
        if b["chord"] != "N.C.":
            key = b["chord"]
            for suffix in ("maj7", "m7", "dim", "sus4", "7", "m"):
                if key.endswith(suffix):
                    key = key[: -len(suffix)] + ("m" if suffix in ("m", "m7") else "")
                    break
            break
    return bpm, key, bars


def bars_to_sections(bars: list[dict]):
    # Practical automatic form guess: 8-bar blocks. Later this can be refined by similarity clustering.
    labels = ["A INTRO / VERSE", "B VERSE / CHORUS", "C BRIDGE", "D OUTRO"]
    sections = []
    for idx in range(0, len(bars), 8):
        block = bars[idx : idx + 8]
        if block:
            sections.append((labels[min(len(sections), len(labels) - 1)], block))
    return sections


def make_abc(title: str, bpm: int, key: str, bars: list[dict]):
    key_clean = FLAT_TO_SHARP.get(key, key).replace("m7", "m").replace("maj7", "").replace("7", "")
    if key_clean not in NOTES and not (len(key_clean) > 1 and key_clean[:-1] in NOTES and key_clean.endswith("m")):
        key_clean = "C"
    lines = [
        "X:1",
        f"T:{title}",
        "M:4/4",
        "L:1/4",
        f"Q:1/4={bpm}",
        f"K:{key_clean}",
    ]
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


def render_modern_daw(stems: dict, title: str):
    tracks = []
    meta = {
        "bass": ["Bass", "🎸", "#a78bfa"],
        "drums": ["Drums", "🥁", "#38bdf8"],
        "vocals": ["Vocals", "🎤", "#f472b6"],
        "other": ["Other", "🎹", "#34d399"],
    }
    for key in ["bass", "drums", "vocals", "other"]:
        path = stems.get(key)
        if path and os.path.exists(path):
            tracks.append({"id": key, "name": meta[key][0], "icon": meta[key][1], "color": meta[key][2], "b64": file_to_b64(path)})
    payload = json.dumps(tracks)
    title_js = json.dumps(title)
    html_doc = """
<!doctype html><html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box} body{margin:0;background:transparent;color:#e5e7eb;font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}.deck{background:linear-gradient(180deg,#111827,#0b1020);border:1px solid rgba(255,255,255,.09);border-radius:22px;padding:18px;box-shadow:0 20px 70px rgba(0,0,0,.34)}.top{display:flex;align-items:center;gap:12px;margin-bottom:14px}.play{width:52px;height:52px;border-radius:50%;border:0;background:linear-gradient(135deg,#7c3aed,#06b6d4);color:white;font-size:18px;font-weight:900;cursor:pointer}.mini{height:34px;min-width:42px;border-radius:10px;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.06);color:#dbeafe;font-weight:800;cursor:pointer}.title{font-weight:800;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font:700 11px ui-monospace,Menlo,monospace;color:#94a3b8;margin-top:3px}.timeline{display:grid;grid-template-columns:48px 1fr 48px;align-items:center;gap:10px;margin:12px 0}.time{font:700 12px ui-monospace,Menlo,monospace;color:#9ca3af}.seek{width:100%;accent-color:#22d3ee}.loopbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;padding:10px;border-radius:14px;background:rgba(255,255,255,.035)}.loopbar label{font:800 11px ui-monospace,Menlo,monospace;color:#94a3b8}.loopbar input{accent-color:#a78bfa}.loopbar .range{width:145px}.lanes{display:flex;flex-direction:column;gap:8px}.lane{display:grid;grid-template-columns:120px 78px 135px 1fr 44px;align-items:center;gap:12px;padding:10px 12px;border:1px solid rgba(255,255,255,.07);border-radius:14px;background:rgba(255,255,255,.035)}.name{font-weight:850}.name span{margin-right:8px}.tog{display:flex;gap:6px}.tog button{height:25px;width:32px;border:0;border-radius:7px;background:#1f2937;color:#9ca3af;font-size:11px;font-weight:900;cursor:pointer}.tog button.active.m{background:#ef4444;color:white}.tog button.active.s{background:#facc15;color:#111827}.vol{width:128px;accent-color:var(--c)}.meter{height:8px;border-radius:20px;background:#020617;overflow:hidden;border:1px solid rgba(255,255,255,.05)}.fill{height:100%;width:0;background:linear-gradient(90deg,var(--c),#fff);opacity:.82;transition:width .08s linear}.db{font:800 11px ui-monospace,Menlo,monospace;color:#94a3b8;text-align:right}@media(max-width:720px){.lane{grid-template-columns:1fr 68px}.vol,.meter,.db{display:none}.timeline{grid-template-columns:40px 1fr 40px}}
</style></head><body><div class="deck"><div class="top"><button id="play" class="play">▶</button><button class="mini" onclick="jump(-10)">↶10</button><button class="mini" onclick="jump(10)">10↷</button><div style="min-width:0;flex:1"><div class="title" id="songTitle"></div><div class="sub">sample-locked multitrack practice mixer • mute/solo/loop/seek/speed</div></div><select id="speed" class="mini" onchange="setSpeed(this.value)"><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option></select></div><div class="timeline"><div class="time" id="cur">00:00</div><input id="seek" class="seek" type="range" min="0" max="1000" value="0"><div class="time" id="dur">00:00</div></div><div class="loopbar"><label><input id="loopOn" type="checkbox"> LOOP</label><label>A <input id="loopA" class="range" type="range" min="0" max="1000" value="0"></label><label>B <input id="loopB" class="range" type="range" min="0" max="1000" value="1000"></label><button class="mini" onclick="setLoopA()">Set A</button><button class="mini" onclick="setLoopB()">Set B</button><button class="mini" onclick="clearLoop()">Clear</button></div><div id="lanes" class="lanes"></div></div>
<script>
const DATA = __TRACKS__; const TITLE = __TITLE__; document.getElementById('songTitle').textContent = TITLE;
let tracks={}, playing=false, raf=null, masterDur=0;
function fmt(t){if(!isFinite(t))return'00:00';let m=Math.floor(t/60),s=Math.floor(t%60);return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')}
function init(){const lanes=document.getElementById('lanes');DATA.forEach(t=>{const a=new Audio('data:audio/wav;base64,'+t.b64);a.preload='auto';tracks[t.id]={...t,audio:a,mute:false,solo:false,vol:1};let row=document.createElement('div');row.className='lane';row.style.setProperty('--c',t.color);row.innerHTML=`<div class="name" style="color:${t.color}"><span>${t.icon}</span>${t.name}</div><div class="tog"><button class="m" id="m_${t.id}" onclick="mute('${t.id}')">M</button><button class="s" id="s_${t.id}" onclick="solo('${t.id}')">S</button></div><input class="vol" id="v_${t.id}" type="range" min="0" max="1.5" step="0.01" value="1" oninput="vol('${t.id}',this.value)"><div class="meter"><div class="fill" id="f_${t.id}"></div></div><div class="db" id="db_${t.id}">0dB</div>`;lanes.appendChild(row);a.onloadedmetadata=()=>{masterDur=Math.max(masterDur,a.duration||0);document.getElementById('dur').textContent=fmt(masterDur)}})}
function all(fn){Object.values(tracks).forEach(t=>fn(t.audio,t))}function syncTo(time){all(a=>{a.currentTime=Math.max(0,Math.min(time,a.duration||time))})}function toggle(){if(!playing){all(a=>a.play());playing=true;document.getElementById('play').textContent='Ⅱ';tick()}else{all(a=>a.pause());playing=false;document.getElementById('play').textContent='▶';cancelAnimationFrame(raf)}}
document.getElementById('play').onclick=toggle;document.getElementById('seek').oninput=e=>syncTo((e.target.value/1000)*masterDur);function jump(s){let t=(Object.values(tracks)[0]?.audio.currentTime||0)+s;syncTo(t)}function setSpeed(v){all(a=>a.playbackRate=parseFloat(v))}function setLoopA(){let t=Object.values(tracks)[0]?.audio.currentTime||0;document.getElementById('loopA').value=Math.round((t/masterDur)*1000)}function setLoopB(){let t=Object.values(tracks)[0]?.audio.currentTime||0;document.getElementById('loopB').value=Math.round((t/masterDur)*1000)}function clearLoop(){document.getElementById('loopOn').checked=false;document.getElementById('loopA').value=0;document.getElementById('loopB').value=1000}function vol(id,v){tracks[id].vol=parseFloat(v);mix();document.getElementById('db_'+id).textContent=Math.round((parseFloat(v)-1)*24)+'dB'}function mute(id){tracks[id].mute=!tracks[id].mute;document.getElementById('m_'+id).classList.toggle('active',tracks[id].mute);mix()}function solo(id){tracks[id].solo=!tracks[id].solo;document.getElementById('s_'+id).classList.toggle('active',tracks[id].solo);mix()}function mix(){let anySolo=Object.values(tracks).some(t=>t.solo);Object.values(tracks).forEach(t=>{let v=t.vol;if(t.mute||(anySolo&&!t.solo))v=0;t.audio.volume=Math.min(1,v);})}function tick(){let lead=Object.values(tracks)[0]?.audio;if(!lead)return;let t=lead.currentTime;document.getElementById('cur').textContent=fmt(t);if(masterDur)document.getElementById('seek').value=Math.round((t/masterDur)*1000);if(document.getElementById('loopOn').checked&&masterDur){let A=document.getElementById('loopA').value/1000*masterDur,B=document.getElementById('loopB').value/1000*masterDur;if(B>A&&t>=B)syncTo(A)}Object.values(tracks).forEach(tr=>{let pulse=(tr.audio.volume||0)*(.25+.75*Math.abs(Math.sin(t*5+(tr.id.length))));document.getElementById('f_'+tr.id).style.width=Math.round(pulse*100)+'%'});raf=requestAnimationFrame(tick)}init();
</script></body></html>
""".replace("__TRACKS__", payload).replace("__TITLE__", title_js)
    components.html(html_doc, height=430)


def render_abc_notation(abc: str, bars: list[dict]):
    abc_json = json.dumps(abc)
    bars_json = json.dumps(bars)
    html_doc = """
<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/abcjs-audio.css">
<script src="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/dist/abcjs-basic-min.js"></script>
<style>
body{margin:0;background:transparent;font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}.paper{background:#fffdf6;color:#101010;border-radius:18px;border:1px solid #d7cfbd;padding:20px;box-shadow:0 18px 60px rgba(0,0,0,.18)}#score svg{width:100%;height:auto}.hint{font:700 12px ui-monospace,Menlo,monospace;color:#746b5a;margin-bottom:10px}.grid{display:grid;grid-template-columns:repeat(8,1fr);gap:6px;margin-top:12px}.cell{border:1px solid #ded6c4;border-radius:9px;padding:7px;text-align:center;background:#fffaf0}.bar{font:700 10px ui-monospace,Menlo,monospace;color:#8a806c}.ch{font-size:18px;font-weight:900;color:#111827}.rt{font-size:10px;color:#6b7280}@media(max-width:820px){.grid{grid-template-columns:repeat(4,1fr)}}
</style></head><body><div class="paper"><div class="hint">abcjs notation engine • source: github.com/paulrosen/abcjs • detected audio chords are rendered as chord symbols over staff rests</div><div id="score"></div><div id="bargrid" class="grid"></div></div><script>
const abc=__ABC__; const bars=__BARS__;
ABCJS.renderAbc('score', abc, {responsive:'resize', staffwidth:900, paddingtop:8, paddingbottom:10, add_classes:true});
const g=document.getElementById('bargrid');bars.forEach(b=>{const d=document.createElement('div');d.className='cell';d.innerHTML=`<div class="bar">${b.bar_num}</div><div class="ch">${String(b.chord).replaceAll('#','♯')}</div><div class="rt">${b.root}</div>`;g.appendChild(d)});
</script></body></html>
""".replace("__ABC__", abc_json).replace("__BARS__", bars_json)
    components.html(html_doc, height=760, scrolling=True)


st.markdown(
    "<div class='app-hero'><div class='brand'>🎸 <span>Bass Studio</span></div><div class='pill'>Demucs FT + abcjs lead sheet</div></div>",
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([4.2, 1.6, 1.4, 1.2])
    with c1:
        query = st.text_input("YouTube", placeholder="YouTube linki veya şarkı adı...", label_visibility="collapsed")
    with c2:
        mode = st.selectbox("Kalite", ["Kaliteli pratik", "Hızlı demo", "Maksimum kalite"], label_visibility="collapsed")
    with c3:
        scope = st.selectbox("Süre", ["İlk 90 sn", "İlk 180 sn", "Tam şarkı"], label_visibility="collapsed")
    with c4:
        run = st.button("Yükle", type="primary", use_container_width=True)
    st.markdown("<div class='small-note'>Kalite için varsayılan: htdemucs_ft + bass/drum reinforcement. Tam şarkı ve maksimum kalite daha uzun sürer.</div>", unsafe_allow_html=True)
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
    process_key = f"{title}|{mode}|{scope}"
    st.subheader(f"{title}")

    if st.session_state.get("process_key") != process_key:
        with st.spinner("Ayrıştırma + akor/notasyon analizi çalışıyor..."):
            stems, sep_settings = separate_stems(audio_path, temp_dir, mode, preview_seconds)
            bpm, key, bars = detect_chords_for_audio(audio_path, duration=preview_seconds or 240)
            st.session_state.stems = stems
            st.session_state.sep_settings = sep_settings
            st.session_state.bpm = bpm
            st.session_state.key = key
            st.session_state.bars = bars
            st.session_state.abc = make_abc(title, bpm, key, bars)
            st.session_state.process_key = process_key

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", st.session_state.sep_settings["model"])
    m2.metric("Shifts", st.session_state.sep_settings["shifts"])
    m3.metric("Tempo", f"{st.session_state.bpm} BPM")
    m4.metric("Ölçü", f"{len(st.session_state.bars)} bars")

    tab1, tab2, tab3 = st.tabs(["🎛 Modern Oynatıcı", "🎼 Notasyon", "🧪 Teknik"])
    with tab1:
        render_modern_daw(st.session_state.stems, title)
    with tab2:
        render_abc_notation(st.session_state.abc, st.session_state.bars)
    with tab3:
        st.code(st.session_state.abc, language="abc")
        st.json(st.session_state.bars[:24])
else:
    st.info("Bir şarkı adı veya YouTube linki girip Yükle'ye basın.")
