import os
import json
import cv2
import numpy as np
import mediapipe as mp
import whisper
import yt_dlp
import torch
import shutil
from groq import Groq
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Inisialisasi
init(autoreset=True)
load_dotenv()  # Load API Key dari file .env

# ==========================================
# KONFIGURASI PENGGUNA
# ==========================================
# Ambil API Key dari environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# URL Video yang ingin diedit
YOUTUBE_URL = "https://www.youtube.com/watch?v=1ziIpehWMiI" 
JUMLAH_KLIP = 3

# Konfigurasi Subtitle
FONT_SIZE = 70
FONT_COLOR = '#FFD700' 
FONT_COLOR_ALT = 'white'
STROKE_COLOR = 'black'
STROKE_WIDTH = 3
# Pastikan font ini ada di sistem kamu, atau ganti dengan 'Arial-Bold'
FONT_TYPE = 'Arial-Bold' 
POSISI_TEKS_Y = 0.75 # 75% dari tinggi video (bisa diatur pixel misal 1100)

# ==========================================
# SETUP PATH & IMAGEMAGICK
# ==========================================
# PENTING: Untuk pengguna Windows, arahkan ke path ImageMagick yang terinstall
# Contoh: change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})
# Jika di Linux/Mac biasanya auto-detect, jika error, uncomment baris bawah:
# change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

TEMP_DIR = "temp"
OUT_DIR = "hasil_shorts"

# Bersihkan folder temp jika perlu, buat folder output
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

client = Groq(api_key=GROQ_API_KEY)

def log_info(msg): print(f"{Fore.CYAN}[INFO] {Style.RESET_ALL}{msg}")
def log_success(msg): print(f"{Fore.GREEN}[SUCCESS] {Style.RESET_ALL}{msg}")
def log_error(msg): print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}{msg}")

# ==========================================
# FUNGSI UTAMA
# ==========================================

def download_video(url):
    output_path = f"{TEMP_DIR}/source_video.mp4"
    # Hapus file lama jika ada
    if os.path.exists(output_path): os.remove(output_path)

    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': f"{TEMP_DIR}/raw_video.%(ext)s",
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        log_info(f"Mendownload Video: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Rename file hasil download ke source_video.mp4
        # yt-dlp kadang menamai file raw_video.mp4 atau raw_video.webm.mp4
        for file in os.listdir(TEMP_DIR):
            if file.startswith("raw_video") and file.endswith(".mp4"):
                shutil.move(os.path.join(TEMP_DIR, file), output_path)
                return True
        return False
    except Exception as e:
        log_error(f"Gagal Download: {e}")
        return False

def transcribe_full(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info(f"Engine Transkripsi berjalan di: {device.upper()}")
    
    try:
        model = whisper.load_model("base", device=device)
        result = model.transcribe(audio_path, language='id', task='transcribe', fp16=False, word_timestamps=True)
        return result
    except Exception as e:
        log_error(f"Error Transkripsi: {e}")
        return None

def analyze_hooks_with_groq(transcript_text, num_clips):
    safe_text = transcript_text[:25000] # Limit token
    log_info(f"Mengirim {len(safe_text)} karakter ke AI Groq...")

    prompt = f"""
    You are a professional Video Editor. Analyze this transcript.
    Find exactly {num_clips} viral segments for TikTok (30-60 seconds each).
    
    CRITERIA:
    1. Must have a strong hook.
    2. Must be self-contained context.
    
    TRANSCRIPT:
    {safe_text} ... (truncated)
    
    OUTPUT STRICT JSON ONLY:
    [
      {{ "start": 120.0, "end": 160.0, "title": "Judul_Klip_1" }},
      {{ "start": 300.5, "end": 350.0, "title": "Judul_Klip_2" }}
    ]
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            response_format={"type": "json_object"},
        )
        result_content = chat_completion.choices[0].message.content
        data = json.loads(result_content)
        
        if isinstance(data, list): return data
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list): return v
        return []
    except Exception as e:
        log_error(f"Groq API Error: {e}")
        return []

def create_hormozi_subtitle(word_data, vid_w, vid_h):
    raw_text = word_data.get('word', word_data.get('text', '')).strip()
    if not raw_text: return None

    text = raw_text.upper()
    color = FONT_COLOR_ALT if len(text) <= 3 else FONT_COLOR

    # Kalkulasi posisi Y (jika float, anggap persentase)
    pos_y = POSISI_TEKS_Y if POSISI_TEKS_Y > 1 else int(vid_h * POSISI_TEKS_Y)

    return (TextClip(
                text,
                fontsize=FONT_SIZE,
                color=color,
                font=FONT_TYPE,
                stroke_color=STROKE_COLOR,
                stroke_width=STROKE_WIDTH,
                method='caption', # Menggunakan method caption agar auto-wrap jika terlalu panjang
                size=(int(vid_w * 0.9), None)
            )
            .set_position(('center', pos_y))
            .set_start(word_data['start'])
            .set_end(word_data['end']))

def process_single_clip(source_video, start_t, end_t, clip_name, segment_words):
    log_info(f"Memproses: {clip_name}")

    try:
        full_clip = VideoFileClip(source_video)
        if end_t > full_clip.duration: end_t = full_clip.duration
        clip = full_clip.subclip(start_t, end_t)

        # 1. Face Tracking & Cropping
        # Simpan sementara untuk dianalisis OpenCV
        temp_sub = f"{TEMP_DIR}/temp_{clip_name}.mp4"
        clip.write_videofile(temp_sub, codec='libx264', audio_codec='aac', logger=None)

        mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
        cap = cv2.VideoCapture(temp_sub)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        centers = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x_c = width // 2
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x_c = int((bbox.xmin + bbox.width/2) * width)
                    break # Ambil wajah pertama saja
            centers.append(x_c)
        cap.release()

        # Smoothing pergerakan kamera
        if not centers: centers = [width//2]
        window = 15
        if len(centers) > window:
            centers = np.convolve(centers, np.ones(window)/window, mode='same')

        def crop_fn(get_frame, t):
            idx = int(t * fps)
            safe_idx = min(idx, len(centers)-1)
            cx = centers[safe_idx]
            img = get_frame(t)
            h, w = img.shape[:2]
            # Rasio 9:16
            target_width = int(h * 9/16)
            
            # Hitung koordinat crop (pastikan tidak keluar batas gambar)
            x1 = int(cx - target_width/2)
            x1 = max(0, min(w - target_width, x1))
            
            return img[:, x1:x1+target_width]

        final_clip = clip.fl(crop_fn, apply_to=['mask']).resize(height=1920) # Resize ke 1080x1920

        # 2. Subtitles
        subs = []
        vid_w, vid_h = final_clip.w, final_clip.h
        valid_words = [w for w in segment_words if w['start'] >= start_t and w['end'] <= end_t]

        for w in valid_words:
            word_data = {
                'word': w.get('word', w.get('text', '')),
                'start': w['start'] - start_t,
                'end': w['end'] - start_t
            }
            try:
                txt_clip = create_hormozi_subtitle(word_data, vid_w, vid_h)
                if txt_clip: subs.append(txt_clip)
            except Exception as e:
                # Kadang error font tidak ditemukan
                print(f"Sub Error: {e}")
                continue

        final = CompositeVideoClip([final_clip] + subs)

        # Output
        safe_name = "".join([c for c in clip_name if c.isalnum() or c=='_'])
        output_filename = f"{OUT_DIR}/{safe_name}.mp4"
        
        # Menggunakan preset ultrafast agar render cepat, threads disesuaikan CPU
        final.write_videofile(output_filename, codec='libx264', audio_codec='aac', fps=24, preset='fast', threads=4, logger=None)
        
        full_clip.close()
        final.close()
        
        # Hapus temp file per klip
        if os.path.exists(temp_sub): os.remove(temp_sub)
        log_success(f"Disimpan: {output_filename}")
        
    except Exception as e:
        log_error(f"Gagal memproses klip {clip_name}: {e}")

def main():
    print(f"\n{Fore.YELLOW}=== AI AUTO SHORTS (LOCAL VERSION) ==={Style.RESET_ALL}\n")

    if not GROQ_API_KEY:
        log_error("API Key Groq tidak ditemukan di file .env!")
        return

    # 1. Download
    if not download_video(YOUTUBE_URL): return
    source_path = f"{TEMP_DIR}/source_video.mp4"

    # 2. Transkrip
    video = VideoFileClip(source_path)
    audio_path = f"{TEMP_DIR}/source_audio.wav"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()

    print("Sedang mentranskripsi audio...")
    whisper_result = transcribe_full(audio_path)
    if not whisper_result: return

    full_text = ""
    for seg in whisper_result['segments']:
        full_text += f"[{seg['start']:.1f}] {seg['text']}\n"
    all_words = [w for seg in whisper_result['segments'] for w in seg['words']]

    # 3. Analisis AI
    print("AI sedang mencari Hooks...")
    clips_data = analyze_hooks_with_groq(full_text, JUMLAH_KLIP)

    if not clips_data:
        log_error("AI tidak menemukan klip.")
        return

    log_success(f"Ditemukan {len(clips_data)} Klip!")
    
    # 4. Proses Editing
    for i, data in enumerate(clips_data):
        print(f"\n🎬 Memproses Klip {i+1}/{len(clips_data)}: {data.get('title')}")
        process_single_clip(
            source_path,
            float(data['start']),
            float(data['end']),
            f"Short_{i+1}_{data.get('title', 'Clip')}",
            all_words
        )

    log_success(f"\nSemua selesai! Cek folder '{OUT_DIR}'")

if __name__ == "__main__":
    main()