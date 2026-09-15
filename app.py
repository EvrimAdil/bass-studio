import os
import time
import base64
import tempfile
import torch
import numpy as np
import librosa
import soundfile as sf
import yt_dlp
import streamlit as st
import streamlit.components.v1 as components
import demucs.api

# Page Config
st.set_page_config(
    page_title="Bass Studio DAW & Real Book",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimalist Studio & Real Book CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; }
    .main .block-container { max-width: 1000px !important; padding-top: 1.2rem; padding-bottom: 3rem; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Title Styling */
    .studio-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .studio-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* REAL BOOK SHEET SCORE STYLING */
    .realbook-container {
        background: #fdfbf7;
        color: #1a1a1a;
        border: 2px solid #2d2d2d;
        border-radius: 8px;
        padding: 24px;
        font-family: 'Caveat', cursive, serif;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        margin-top: 1.5rem;
    }

    .realbook-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
        border-bottom: 2px solid #1a1a1a;
        padding-bottom: 6px;
    }

    .realbook-meta {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 18px;
        color: #333;
    }

    .realbook-section-label {
        font-size: 1.6rem;
        font-weight: 700;
        background: #1a1a1a;
        color: #fdfbf7;
        padding: 2px 10px;
        border-radius: 4px;
        display: inline-block;
        margin-top: 14px;
        margin-bottom: 10px;
    }

    .realbook-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        border: 2px solid #1a1a1a;
        background: #fdfbf7;
        margin-bottom: 16px;
    }

    .realbook-measure {
        border-right: 2px solid #1a1a1a;
        padding: 12px 8px;
        text-align: center;
        min-height: 70px;
        position: relative;
    }

    .realbook-measure:last-child {
        border-right: none;
    }

    .realbook-chord {
        font-size: 2rem;
        font-weight: 700;
        color: #000;
        line-height: 1;
    }

    .realbook-bass-note {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #666;
        margin-top: 6px;
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_demucs_model():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return demucs.api.Separator(model='htdemucs', device=device, shifts=1)

def separate_stems_fast(audio_path, output_dir, max_duration_sec=60):
    separator = load_demucs_model()
    
    if max_duration_sec:
        y, sr = librosa.load(audio_path, sr=44100, duration=max_duration_sec)
        trimmed_path = os.path.join(output_dir, "trimmed_input.wav")
        sf.write(trimmed_path, y, sr)
        audio_path = trimmed_path

    origin, res = separator.separate_audio_file(audio_path)
    
    stem_paths = {}
    for stem_name, source in res.items():
        out_file = os.path.join(output_dir, f"{stem_name}.wav")
        demucs.api.save_audio(source, out_file, samplerate=separator.samplerate)
        stem_paths[stem_name] = out_file
        
    return stem_paths

def download_youtube_audio(query_or_url, output_dir):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, 'yt_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'default_search': 'ytsearch1:'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_or_url, download=True)
        if 'entries' in info and len(info['entries']) > 0:
            info = info['entries'][0]
        title = info.get('title', 'YouTube Song')
        
    out_file = os.path.join(output_dir, "yt_audio.mp3")
    return title, out_file

def analyze_track(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=45)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(np.round(float(tempo))) if tempo else 120

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_idx = int(np.argmax(chroma_avg))
        estimated_key = notes[root_idx]

        return {
            "bpm": bpm,
            "key": f"{estimated_key} Minor",
            "time_sig": "4/4",
            "scale": f"{estimated_key} Pentatonic / {estimated_key} Dorian"
        }
    except Exception:
        return {"bpm": 120, "key": "A Minor", "time_sig": "4/4", "scale": "A Pentatonic Minor"}

def file_to_b64(filepath):
    """Convert audio file to base64 for inline Web Audio API DAW player."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# DAW Web Audio API Player Component
def render_daw_mixer(stems_dict):
    b64_stems = {}
    for name, path in stems_dict.items():
        if os.path.exists(path):
            b64_stems[name] = file_to_b64(path)

    daw_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 12px;
        }}
        .daw-container {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
        }}
        .transport-bar {{
            display: flex;
            align-items: center;
            gap: 16px;
            background: #1e293b;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
        }}
        .btn {{
            background: #3b82f6;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
        }}
        .btn:hover {{ opacity: 0.9; }}
        .btn-stop {{ background: #ef4444; }}
        .track-lane {{
            display: grid;
            grid-template-columns: 140px 80px 1fr;
            align-items: center;
            gap: 16px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
        }}
        .track-name {{ font-weight: 700; font-size: 0.9rem; }}
        .bass-name {{ color: #a855f7; }}
        .drums-name {{ color: #3b82f6; }}
        .vocals-name {{ color: #ec4899; }}
        .other-name {{ color: #10b981; }}
        
        .btn-m {{ background: #334155; color: #94a3b8; border: none; padding: 4px 8px; border-radius: 4px; font-weight: 700; cursor: pointer; }}
        .btn-m.active {{ background: #ef4444; color: white; }}
        .btn-s {{ background: #334155; color: #94a3b8; border: none; padding: 4px 8px; border-radius: 4px; font-weight: 700; cursor: pointer; }}
        .btn-s.active {{ background: #eab308; color: black; }}

        input[type=range] {{ width: 100%; accent-color: #3b82f6; }}
    </style>
    </head>
    <body>
    <div class="daw-container">
        <div class="transport-bar">
            <button id="playBtn" class="btn" onclick="togglePlay()">▶ PLAY</button>
            <button id="stopBtn" class="btn btn-stop" onclick="stopAudio()">⏹ STOP</button>
            <span id="timeDisplay" style="font-family: monospace; font-weight: 700;">00:00</span>
            <input type="range" id="seekBar" value="0" min="0" max="100" step="0.1" oninput="seekAudio(this.value)">
        </div>

        <!-- Bass Track -->
        <div class="track-lane">
            <div class="track-name bass-accent">🎸 BASS</div>
            <div>
                <button id="m_bass" class="btn-m" onclick="toggleMute('bass')">M</button>
                <button id="s_bass" class="btn-s" onclick="toggleSolo('bass')">S</button>
            </div>
            <input type="range" id="v_bass" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('bass', this.value)">
        </div>

        <!-- Drums Track -->
        <div class="track-lane">
            <div class="track-name drums-accent">🥁 DRUMS</div>
            <div>
                <button id="m_drums" class="btn-m" onclick="toggleMute('drums')">M</button>
                <button id="s_drums" class="btn-s" onclick="toggleSolo('drums')">S</button>
            </div>
            <input type="range" id="v_drums" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('drums', this.value)">
        </div>

        <!-- Vocals Track -->
        <div class="track-lane">
            <div class="track-name vocals-accent">🎤 VOCALS</div>
            <div>
                <button id="m_vocals" class="btn-m" onclick="toggleMute('vocals')">M</button>
                <button id="s_vocals" class="btn-s" onclick="toggleSolo('vocals')">S</button>
            </div>
            <input type="range" id="v_vocals" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('vocals', this.value)">
        </div>

        <!-- Other Track -->
        <div class="track-lane">
            <div class="track-name other-accent">🎹 OTHER</div>
            <div>
                <button id="m_other" class="btn-m" onclick="toggleMute('other')">M</button>
                <button id="s_other" class="btn-s" onclick="toggleSolo('other')">S</button>
            </div>
            <input type="range" id="v_other" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('other', this.value)">
        </div>
    </div>

    <script>
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const tracks = {{
            bass: {{ b64: "{b64_stems.get('bass', '')}", gainNode: null, audio: null, mute: false, solo: false }},
            drums: {{ b64: "{b64_stems.get('drums', '')}", gainNode: null, audio: null, mute: false, solo: false }},
            vocals: {{ b64: "{b64_stems.get('vocals', '')}", gainNode: null, audio: null, mute: false, solo: false }},
            other: {{ b64: "{b64_stems.get('other', '')}", gainNode: null, audio: null, mute: false, solo: false }}
        }};

        let isPlaying = false;

        // Initialize Audio Elements
        Object.keys(tracks).forEach(key => {{
            if (tracks[key].b64) {{
                const audio = new Audio("data:audio/wav;base64," + tracks[key].b64);
                const source = audioCtx.createMediaElementSource(audio);
                const gainNode = audioCtx.createGain();
                source.connect(gainNode);
                gainNode.connect(audioCtx.destination);
                tracks[key].audio = audio;
                tracks[key].gainNode = gainNode;
            }}
        }});

        function togglePlay() {{
            if (audioCtx.state === 'suspended') {{ audioCtx.resume(); }}
            const btn = document.getElementById("playBtn");
            if (!isPlaying) {{
                Object.keys(tracks).forEach(key => {{
                    if (tracks[key].audio) tracks[key].audio.play();
                }});
                isPlaying = true;
                btn.innerText = "⏸ PAUSE";
            }} else {{
                Object.keys(tracks).forEach(key => {{
                    if (tracks[key].audio) tracks[key].audio.pause();
                }});
                isPlaying = false;
                btn.innerText = "▶ PLAY";
            }}
        }}

        function stopAudio() {{
            Object.keys(tracks).forEach(key => {{
                if (tracks[key].audio) {{
                    tracks[key].audio.pause();
                    tracks[key].audio.currentTime = 0;
                }}
            }});
            isPlaying = false;
            document.getElementById("playBtn").innerText = "▶ PLAY";
        }}

        function setVolume(trackName, val) {{
            if (tracks[trackName] && tracks[trackName].gainNode) {{
                tracks[trackName].gainNode.gain.value = parseFloat(val);
            }}
        }}

        function toggleMute(trackName) {{
            tracks[trackName].mute = !tracks[trackName].mute;
            const btn = document.getElementById("m_" + trackName);
            btn.classList.toggle("active", tracks[trackName].mute);
            updateMix();
        }}

        function toggleSolo(trackName) {{
            tracks[trackName].solo = !tracks[trackName].solo;
            const btn = document.getElementById("s_" + trackName);
            btn.classList.toggle("active", tracks[trackName].solo);
            updateMix();
        }}

        function updateMix() {{
            const anySolo = Object.keys(tracks).some(k => tracks[k].solo);
            Object.keys(tracks).forEach(k => {{
                if (tracks[k].gainNode) {{
                    let vol = parseFloat(document.getElementById("v_" + k).value);
                    if (tracks[k].mute) {{ vol = 0; }}
                    else if (anySolo && !tracks[k].solo) {{ vol = 0; }}
                    tracks[k].gainNode.gain.value = vol;
                }}
            }});
        }}
    </script>
    </body>
    </html>
    """
    components.html(daw_html, height=340)

# Main Studio Header
st.markdown('<div class="studio-header"><div class="studio-title">🎸 Bass Studio DAW & Real Book</div></div>', unsafe_allow_html=True)

# Search Bar
col_s, col_b = st.columns([4, 1])
with col_s:
    yt_query = st.text_input("Arama", placeholder="YouTube Linki veya Şarkı Adı yazın... (örn: Chili Peppers Can't Stop, Duman Seni Kendime Sakladım)", label_visibility="collapsed")
with col_b:
    fetch_btn = st.button("🔴 Şarkıyı Yükle", type="primary", use_container_width=True)

if fetch_btn and yt_query:
    temp_dir = tempfile.mkdtemp()
    with st.spinner("📥 YouTube'dan ses çekiliyor..."):
        try:
            song_title, target_path = download_youtube_audio(yt_query, temp_dir)
            st.session_state.current_title = song_title
            st.session_state.current_path = target_path
            st.session_state.temp_dir = temp_dir
        except Exception as e:
            st.error(f"Hata: {e}")

if "current_path" in st.session_state and os.path.exists(st.session_state.current_path):
    audio_path = st.session_state.current_path
    title = st.session_state.current_title
    temp_dir = st.session_state.temp_dir

    st.subheader(f"🎵 {title}")

    # Stem Separation Trigger
    if "stems" not in st.session_state or st.session_state.get("active_title") != title:
        with st.spinner("⚡ Apple Silicon GPU hızlandırması ile 4 kanallı DAW stemleri üretiliyor..."):
            stems = separate_stems_fast(audio_path, temp_dir, max_duration_sec=60)
            analysis = analyze_track(audio_path)
            st.session_state.stems = stems
            st.session_state.analysis = analysis
            st.session_state.active_title = title

    stems = st.session_state.stems
    analysis = st.session_state.analysis

    # 1. DAW MULTITRACK MIXER
    st.markdown("### 🎛️ Multitrack DAW Mixer")
    render_daw_mixer(stems)

    st.markdown("---")

    # 2. REAL BOOK LEAD SHEET SCORE VIEW
    st.markdown("### 🎼 Real Book Lead Sheet Score")

    st.markdown(f"""
    <div class="realbook-container">
        <div class="realbook-title">{title}</div>
        <div class="realbook-meta">
            <span>KEY: {analysis['key']}</span>
            <span>TEMPO: {analysis['bpm']} BPM</span>
            <span>TIME: {analysis['time_sig']}</span>
            <span>STYLE: Funk / Rock Bass</span>
        </div>

        <!-- Section A: Verse -->
        <div class="realbook-section-label">[A] VERSE</div>
        <div class="realbook-grid">
            <div class="realbook-measure">
                <div class="realbook-chord">Am7</div>
                <div class="realbook-bass-note">Root: A (La)</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">D7</div>
                <div class="realbook-bass-note">Root: D (Re)</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">Gmaj7</div>
                <div class="realbook-bass-note">Root: G (Sol)</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">Cmaj7</div>
                <div class="realbook-bass-note">Root: C (Do)</div>
            </div>
        </div>

        <!-- Section B: Chorus -->
        <div class="realbook-section-label">[B] CHORUS</div>
        <div class="realbook-grid">
            <div class="realbook-measure">
                <div class="realbook-chord">Fmaj7</div>
                <div class="realbook-bass-note">Root: F (Fa)</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">Bm7b5</div>
                <div class="realbook-bass-note">Root: B (Si)</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">E7alt</div>
                <div class="realbook-bass-note">Root: E (Mi)</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">Am7</div>
                <div class="realbook-bass-note">Root: A (La)</div>
            </div>
        </div>

        <!-- Section C: Bass Solo / Interlude -->
        <div class="realbook-section-label">[C] BASS SOLO & RIFF</div>
        <div class="realbook-grid">
            <div class="realbook-measure">
                <div class="realbook-chord">Am (Slap)</div>
                <div class="realbook-bass-note">Groove: A1-A2 Octave</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">Em7</div>
                <div class="realbook-bass-note">Walk: E-G-A-A#</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">F7</div>
                <div class="realbook-bass-note">Walk: F-A-C-Eb</div>
            </div>
            <div class="realbook-measure">
                <div class="realbook-chord">E7(#9)</div>
                <div class="realbook-bass-note">Fill: E-G#-B-D</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Başlamak için yukarıya bir YouTube şarkı linki veya adı yazın.")
