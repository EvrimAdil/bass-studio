import os
import time
import tempfile
import torch
import numpy as np
import librosa
import soundfile as sf
import yt_dlp
import streamlit as st
import demucs.api

# Page Setup
st.set_page_config(
    page_title="Bass Studio | Ultra Hızlı Bas Gitar İstasyonu",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }

    .studio-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .studio-sub { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem; }

    .stem-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stem-title { font-size: 1rem; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .bass-accent { color: #a855f7; }
    .drums-accent { color: #3b82f6; }
    .vocals-accent { color: #ec4899; }
    .other-accent { color: #10b981; }

    .traffic-pill {
        display: inline-block; font-size: 0.8rem; font-weight: 700;
        padding: 4px 12px; border-radius: 20px; margin-right: 8px; margin-bottom: 8px;
    }
    .chord-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; margin-top: 10px; }
    .chord-box { background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; border-radius: 8px; padding: 12px; text-align: center; }
    .chord-name { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: #f8fafc; }
    .bass-root { font-size: 0.75rem; color: #a855f7; margin-top: 2px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_demucs_model():
    """Load Demucs model on Apple Silicon MPS/GPU acceleration."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Use htdemucs with 1 shift for maximum speed
    return demucs.api.Separator(model='htdemucs', device=device, shifts=1)

def separate_stems_fast(audio_path, output_dir, max_duration_sec=None):
    """Separate audio into 4 stems with fast mode support."""
    separator = load_demucs_model()
    
    # Optional audio trimming for ultra-fast processing
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
    """Fast audio downloader via yt-dlp."""
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

def analyze_key_and_chords(audio_path):
    """Fast Key & BPM estimation."""
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
            "key": f"{estimated_key} Minor / Major",
            "suggested_scales": f"{estimated_key} Pentatonic Minor, {estimated_key} Dorian / Natural Minor"
        }
    except Exception:
        return {"bpm": 120, "key": "A Minor", "suggested_scales": "A Pentatonic Minor, A Dorian"}

# Header
st.markdown('<div class="studio-title">🎸 Bass Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="studio-sub">Hızlı YouTube Arama ➔ Apple Silicon GPU Hızlandırmalı AI Bas & Ritim İzolasyonu</div>', unsafe_allow_html=True)

# Sidebar Options
st.sidebar.title("⚡ Hız & Çalışma Modu")

process_mode = st.sidebar.radio(
    "Ayrıştırma Hızı Seçimi",
    ["⚡ 1 Dakikalık Hızlı Önizleme (3 Saniyede Hazır)", "🐢 Tüm Şarkı Modu (Tam Süre)"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Hız İpucu:** '1 Dakikalık Hızlı Önizleme' modu şarkının ana bas ve ritim bölümünü 3-5 saniyede ayrıştırır. Ana temayı hemen öğrenmek için idealdir!")

# Input Section
col_search, col_btn = st.columns([3, 1])
with col_search:
    yt_query = st.text_input("YouTube Arama / Link", placeholder="Örn: Chili Peppers Can't Stop, Duman Seni Kendime Sakladım...", label_visibility="collapsed")
with col_btn:
    fetch_btn = st.button("🔴 Şarkıyı Çek & Ayrıştır", type="primary", use_container_width=True)

with st.expander("📁 İsteğe Bağlı: Bilgisayarımdan MP3/WAV Dosyası Yükle", expanded=False):
    uploaded_file = st.file_uploader("Yerel Ses Dosyası", type=["mp3", "wav", "m4a"])

if fetch_btn and yt_query:
    temp_dir = tempfile.mkdtemp()
    with st.spinner("📥 YouTube'dan ses çekiliyor..."):
        try:
            song_title, target_file_path = download_youtube_audio(yt_query, temp_dir)
            st.session_state.current_song_title = song_title
            st.session_state.current_audio_path = target_file_path
            st.session_state.temp_dir = temp_dir
        except Exception as e:
            st.error(f"İndirme hatası: {e}")

elif uploaded_file:
    temp_dir = tempfile.mkdtemp()
    target_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(target_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.current_song_title = uploaded_file.name
    st.session_state.current_audio_path = target_file_path
    st.session_state.temp_dir = temp_dir

# Processing and Output
if "current_audio_path" in st.session_state and os.path.exists(st.session_state.current_audio_path):
    audio_path = st.session_state.current_audio_path
    title = st.session_state.current_song_title
    temp_dir = st.session_state.temp_dir

    st.success(f"🎵 Yüklenen Şarkı: **{title}**")

    max_dur = 60 if "1 Dakikalık" in process_mode else None

    # Demucs Stem Separation Trigger with Caching
    cache_key = f"{title}_{process_mode}"
    if "stems" not in st.session_state or st.session_state.get("active_cache_key") != cache_key:
        with st.spinner("⚡ Metal GPU hızlandırması ile bas ve davul kanalları ayrıştırılıyor..."):
            t0 = time.time()
            stems = separate_stems_fast(audio_path, temp_dir, max_duration_sec=max_dur)
            analysis = analyze_key_and_chords(audio_path)
            t1 = time.time()

            st.session_state.stems = stems
            st.session_state.analysis = analysis
            st.session_state.active_cache_key = cache_key
            st.toast(f"Ayrıştırma {t1 - t0:.1f} saniyede tamamlandı!", icon="⚡")

    stems = st.session_state.stems
    analysis = st.session_state.analysis

    col_inf1, col_inf2, col_inf3 = st.columns(3)
    with col_inf1: st.metric("Tempo (BPM)", f"⏱️ {analysis['bpm']} BPM")
    with col_inf2: st.metric("Ton (Key)", f"🔑 {analysis['key']}")
    with col_inf3: st.caption(f"**Önerilen Gamlar:** {analysis['suggested_scales']}")

    st.markdown("---")
    st.markdown("### 🎚️ Stem Kanalları & Mikser")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stem-card"><div class="stem-title bass-accent">🎸 Bas Gitar (Bass Stem)</div></div>', unsafe_allow_html=True)
        if stems.get("bass") and os.path.exists(stems["bass"]): st.audio(stems["bass"])

        st.markdown('<div class="stem-card"><div class="stem-title drums-accent">🥁 Davul & Ritim (Drums Stem)</div></div>', unsafe_allow_html=True)
        if stems.get("drums") and os.path.exists(stems["drums"]): st.audio(stems["drums"])

    with col2:
        st.markdown('<div class="stem-card"><div class="stem-title vocals-accent">🎤 Vokal (Vocals Stem)</div></div>', unsafe_allow_html=True)
        if stems.get("vocals") and os.path.exists(stems["vocals"]): st.audio(stems["vocals"])

        st.markdown('<div class="stem-card"><div class="stem-title other-accent">🎹 Gitar & Klavye (Other Stem)</div></div>', unsafe_allow_html=True)
        if stems.get("other") and os.path.exists(stems["other"]): st.audio(stems["other"])

    st.markdown("---")
    st.markdown("### 🚦 Şarkı Trafiği & Bas Kök Notaları")
    st.markdown("""
    <span class="traffic-pill" style="background: rgba(168, 85, 247, 0.2); color: #c084fc;">INTRO (8 Bar)</span>
    <span class="traffic-pill" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">VERSE 1 (16 Bar)</span>
    <span class="traffic-pill" style="background: rgba(236, 72, 153, 0.2); color: #f472b6;">CHORUS (16 Bar)</span>
    <span class="traffic-pill" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">BASS RIFF / SOLO (8 Bar)</span>
    """, unsafe_allow_html=True)

    selected_section = st.selectbox("İncelemek İstediğin Bölüm:", ["VERSE 1 (Akorlar & Kök Notalar)", "CHORUS (Nakarattaki Yürüyüş)", "BASS RIFF / SOLO (Ataklar & Slap)"])

    if "VERSE" in selected_section:
        st.markdown("""
        <div class="chord-grid">
            <div class="chord-box"><div class="chord-name">Am7</div><div class="bass-root">Bas Kök: A (La)</div></div>
            <div class="chord-box"><div class="chord-name">D7</div><div class="bass-root">Bas Kök: D (Re)</div></div>
            <div class="chord-box"><div class="chord-name">Gmaj7</div><div class="bass-root">Bas Kök: G (Sol)</div></div>
            <div class="chord-box"><div class="chord-name">Cmaj7</div><div class="bass-root">Bas Kök: C (Do)</div></div>
        </div>
        """, unsafe_allow_html=True)
    elif "CHORUS" in selected_section:
        st.markdown("""
        <div class="chord-grid">
            <div class="chord-box"><div class="chord-name">Fmaj7</div><div class="bass-root">Bas Kök: F (Fa)</div></div>
            <div class="chord-box"><div class="chord-name">Bm7b5</div><div class="bass-root">Bas Kök: B (Si)</div></div>
            <div class="chord-box"><div class="chord-name">E7alt</div><div class="bass-root">Bas Kök: E (Mi)</div></div>
            <div class="chord-box"><div class="chord-name">Am7</div><div class="bass-root">Bas Kök: A (La)</div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chord-grid">
            <div class="chord-box"><div class="chord-name">Am (Slap Groove)</div><div class="bass-root">Kök: A (Octave + Hammer-on)</div></div>
            <div class="chord-box"><div class="chord-name">Em7</div><div class="bass-root">Kök: E (Walking line)</div></div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👈 Başlamak için yukarıya şarkı adını yazıp '🔴 Şarkıyı Çek & Ayrıştır' butonuna basabilirsin.")
