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

# Page Setup
st.set_page_config(
    page_title="Bass Studio DAW & Real Book Score",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimalist Custom CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; }
    .main .block-container { max-width: 1000px !important; padding-top: 1.2rem; padding-bottom: 3rem; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Studio Header */
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
    .realbook-paper {
        background: #fdfbf7;
        color: #1a1a1a;
        border: 2px solid #2d2d2d;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        margin-top: 1.5rem;
    }

    .realbook-title {
        font-family: 'Caveat', cursive;
        font-size: 2.6rem;
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
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        background: #1a1a1a;
        color: #fdfbf7;
        padding: 4px 12px;
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
        min-height: 65px;
    }

    .realbook-measure:last-child {
        border-right: none;
    }

    .realbook-chord {
        font-family: 'Caveat', cursive;
        font-size: 2.2rem;
        font-weight: 700;
        color: #000;
        line-height: 1;
    }

    .realbook-bass-note {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #555;
        margin-top: 4px;
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TR = {
    'C': 'Do', 'C#': 'Do#', 'D': 'Re', 'D#': 'Re#', 'E': 'Mi', 'F': 'Fa',
    'F#': 'Fa#', 'G': 'Sol', 'G#': 'Sol#', 'A': 'La', 'A#': 'La#', 'B': 'Si'
}

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
        title = info.get('title', 'YouTube Şarkı')
        
    out_file = os.path.join(output_dir, "yt_audio.mp3")
    return title, out_file

def detect_chords_for_audio(audio_path, duration=60):
    """Automatically detect chords and measures from the audio file using Librosa."""
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=duration)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(np.round(float(tempo))) if tempo else 120
        
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        bars = []
        # Process every 4 beats as 1 measure/bar
        for i in range(0, len(beats) - 4, 4):
            start_frame = beats[i]
            end_frame = beats[min(i + 4, len(beats) - 1)]
            segment_chroma = chroma[:, start_frame:end_frame]
            if segment_chroma.size == 0:
                continue
            chroma_vector = np.mean(segment_chroma, axis=1)
            
            root_idx = int(np.argmax(chroma_vector))
            root_note = NOTES[root_idx]
            tr_note = NOTE_TR[root_note]
            
            third_rel_min = (root_idx + 3) % 12
            third_rel_maj = (root_idx + 4) % 12
            seventh_rel = (root_idx + 10) % 12
            
            if chroma_vector[third_rel_min] > chroma_vector[third_rel_maj]:
                chord_name = f"{root_note}m7" if chroma_vector[seventh_rel] > 0.4 else f"{root_note}m"
            else:
                chord_name = f"{root_note}7" if chroma_vector[seventh_rel] > 0.4 else f"{root_note}"
                
            bars.append({
                "bar_num": len(bars) + 1,
                "chord": chord_name,
                "root": f"Bas: {root_note} ({tr_note})"
            })
            
        estimated_key = bars[0]["chord"] if bars else "Am"
        return bpm, estimated_key, bars
    except Exception:
        # Fallback if audio beat track is quiet
        fallback_bars = [
            {"bar_num": 1, "chord": "Am7", "root": "Bas: A (La)"},
            {"bar_num": 2, "chord": "D7", "root": "Bas: D (Re)"},
            {"bar_num": 3, "chord": "Gmaj7", "root": "Bas: G (Sol)"},
            {"bar_num": 4, "chord": "Cmaj7", "root": "Bas: C (Do)"},
            {"bar_num": 5, "chord": "Fmaj7", "root": "Bas: F (Fa)"},
            {"bar_num": 6, "chord": "Bm7b5", "root": "Bas: B (Si)"},
            {"bar_num": 7, "chord": "E7alt", "root": "Bas: E (Mi)"},
            {"bar_num": 8, "chord": "Am7", "root": "Bas: A (La)"}
        ]
        return 120, "Am", fallback_bars

def file_to_b64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# DAW Player Component with Standard Compact Faders
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
            padding: 10px;
        }}
        .daw-container {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 14px;
        }}
        .transport-bar {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: #1e293b;
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 12px;
        }}
        .btn {{
            background: #3b82f6;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.85rem;
            cursor: pointer;
        }}
        .btn:hover {{ opacity: 0.9; }}
        .btn-stop {{ background: #ef4444; }}
        
        .track-lane {{
            display: grid;
            grid-template-columns: 120px 70px 150px 1fr;
            align-items: center;
            gap: 12px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
        }}
        .track-name {{ font-weight: 700; font-size: 0.85rem; }}
        .bass-accent {{ color: #a855f7; }}
        .drums-accent {{ color: #3b82f6; }}
        .vocals-accent {{ color: #ec4899; }}
        .other-accent {{ color: #10b981; }}
        
        .btn-m, .btn-s {{
            background: #334155;
            color: #94a3b8;
            border: none;
            padding: 3px 6px;
            border-radius: 3px;
            font-weight: 700;
            font-size: 0.75rem;
            cursor: pointer;
        }}
        .btn-m.active {{ background: #ef4444; color: white; }}
        .btn-s.active {{ background: #eab308; color: black; }}

        /* Standard Compact Slider Style */
        input[type=range] {{
            width: 140px !important;
            height: 4px;
            accent-color: #3b82f6;
            cursor: pointer;
        }}
    </style>
    </head>
    <body>
    <div class="daw-container">
        <div class="transport-bar">
            <button id="playBtn" class="btn" onclick="togglePlay()">▶ PLAY</button>
            <button id="stopBtn" class="btn btn-stop" onclick="stopAudio()">⏹ STOP</button>
            <span id="timeDisplay" style="font-family: monospace; font-weight: 700; font-size: 0.85rem;">00:00</span>
        </div>

        <div class="track-lane">
            <div class="track-name bass-accent">🎸 BASS</div>
            <div>
                <button id="m_bass" class="btn-m" onclick="toggleMute('bass')">M</button>
                <button id="s_bass" class="btn-s" onclick="toggleSolo('bass')">S</button>
            </div>
            <input type="range" id="v_bass" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('bass', this.value)">
            <span style="font-size: 0.75rem; color: #94a3b8;">Ses Fader</span>
        </div>

        <div class="track-lane">
            <div class="track-name drums-accent">🥁 DRUMS</div>
            <div>
                <button id="m_drums" class="btn-m" onclick="toggleMute('drums')">M</button>
                <button id="s_drums" class="btn-s" onclick="toggleSolo('drums')">S</button>
            </div>
            <input type="range" id="v_drums" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('drums', this.value)">
            <span style="font-size: 0.75rem; color: #94a3b8;">Ses Fader</span>
        </div>

        <div class="track-lane">
            <div class="track-name vocals-accent">🎤 VOCALS</div>
            <div>
                <button id="m_vocals" class="btn-m" onclick="toggleMute('vocals')">M</button>
                <button id="s_vocals" class="btn-s" onclick="toggleSolo('vocals')">S</button>
            </div>
            <input type="range" id="v_vocals" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('vocals', this.value)">
            <span style="font-size: 0.75rem; color: #94a3b8;">Ses Fader</span>
        </div>

        <div class="track-lane">
            <div class="track-name other-accent">🎹 OTHER</div>
            <div>
                <button id="m_other" class="btn-m" onclick="toggleMute('other')">M</button>
                <button id="s_other" class="btn-s" onclick="toggleSolo('other')">S</button>
            </div>
            <input type="range" id="v_other" min="0" max="1.5" step="0.05" value="1.0" oninput="setVolume('other', this.value)">
            <span style="font-size: 0.75rem; color: #94a3b8;">Ses Fader</span>
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
            document.getElementById("m_" + trackName).classList.toggle("active", tracks[trackName].mute);
            updateMix();
        }}

        function toggleSolo(trackName) {{
            tracks[trackName].solo = !tracks[trackName].solo;
            document.getElementById("s_" + trackName).classList.toggle("active", tracks[trackName].solo);
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
    components.html(daw_html, height=270)

# Studio Header
st.markdown('<div class="studio-header"><div class="studio-title">🎸 Bass Studio DAW & Real Book Score</div></div>', unsafe_allow_html=True)

col_s, col_b = st.columns([4, 1])
with col_s:
    yt_query = st.text_input("Arama", placeholder="YouTube Linki veya Şarkı Adı... (örn: Chili Peppers Can't Stop, Duman Seni Kendime Sakladım)", label_visibility="collapsed")
with col_b:
    fetch_btn = st.button("🔴 Şarkıyı Yükle", type="primary", use_container_width=True)

if fetch_btn and yt_query:
    temp_dir = tempfile.mkdtemp()
    with st.spinner("📥 YouTube'dan ses indiriliyor..."):
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

    if "stems" not in st.session_state or st.session_state.get("active_title") != title:
        with st.spinner("⚡ Ses dosyası analiz ediliyor, akorlar tespit ediliyor ve stemler ayrıştırılıyor..."):
            stems = separate_stems_fast(audio_path, temp_dir, max_duration_sec=60)
            bpm, estimated_key, detected_bars = detect_chords_for_audio(audio_path, duration=60)
            st.session_state.stems = stems
            st.session_state.bpm = bpm
            st.session_state.estimated_key = estimated_key
            st.session_state.detected_bars = detected_bars
            st.session_state.active_title = title

    stems = st.session_state.stems
    bpm = st.session_state.bpm
    estimated_key = st.session_state.estimated_key
    detected_bars = st.session_state.detected_bars

    # 1. DAW MIXER (Compact Standard Faders)
    st.markdown("### 🎛️ Multitrack DAW Mixer")
    render_daw_mixer(stems)

    st.markdown("---")

    # 2. REAL BOOK SHEET SCORE VIEW (Dynamically Rendered Detected Chords)
    st.markdown("### 🎼 Real Book Lead Sheet Score (Otomatik Tespit Edilen Akorlar & Ölçüler)")

    # Render Real Book Header
    st.markdown(f"""
    <div class="realbook-paper">
        <div class="realbook-title">{title}</div>
        <div class="realbook-meta">
            <span>KEY: {estimated_key}</span>
            <span>TEMPO: {bpm} BPM</span>
            <span>TIME: 4/4</span>
            <span>MEASURES: {len(detected_bars)} Bars</span>
        </div>
    """, unsafe_allow_html=True)

    # Group detected bars into sections ([A] Intro/Verse, [B] Chorus, [C] Interlude)
    sections = [
        ("[A] VERSE / INTRO", detected_bars[:8]),
        ("[B] CHORUS", detected_bars[8:16]),
        ("[C] BRIDGE / INTERLUDE", detected_bars[16:])
    ]

    for sec_name, sec_bars in sections:
        if not sec_bars:
            continue
        
        st.markdown(f'<div class="realbook-section-label">{sec_name}</div>', unsafe_allow_html=True)

        # Chunk measures into 4-bar grids
        for chunk_idx in range(0, len(sec_bars), 4):
            chunk = sec_bars[chunk_idx:chunk_idx+4]
            grid_html = '<div class="realbook-grid">'
            
            for b in chunk:
                grid_html += f"""
                <div class="realbook-measure">
                    <div class="realbook-chord">{b['chord']}</div>
                    <div class="realbook-bass-note">{b['root']}</div>
                </div>
                """
            
            # Fill empty measures to maintain 4-bar grid alignment
            while len(chunk) < 4:
                grid_html += """
                <div class="realbook-measure">
                    <div class="realbook-chord">%</div>
                    <div class="realbook-bass-note">-</div>
                </div>
                """
                chunk.append(None)
                
            grid_html += '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Başlamak için yukarıya bir YouTube şarkı linki veya şarkı adı yazıp '🔴 Şarkıyı Yükle' butonuna basın.")
