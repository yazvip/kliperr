# AI Auto Shorts (Local Version)

An automated tool to create viral short-form content (TikTok, Reels, Shorts) from YouTube videos using AI. This tool downloads a video, transcribes it, identifies viral hooks, crops it to 9:16 format with face tracking, and adds dynamic "Hormozi-style" subtitles.

## 🚀 Features

- **Video Downloader**: Automatically downloads YouTube videos in optimized quality.
- **AI Transcription**: Uses OpenAI's Whisper (local) for accurate speech-to-text with word-level timestamps.
- **Hook Analysis**: Leverages Groq (Llama 3.3 70B) to identify the most engaging segments.
- **Auto-Face Tracking**: Dynamically crops videos to 9:16 format while keeping the speaker in focus using MediaPipe.
- **Dynamic Subtitles**: Generates stylish, colorful subtitles inspired by Alex Hormozi's content style.
- **Multi-Threading**: Optimized rendering using MoviePy and OpenCV.

## 🛠️ Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python 3.8+**
2.  **FFmpeg**: Required for MoviePy and Whisper.
3.  **ImageMagick**: Required for `TextClip` in MoviePy.
    - *Windows users*: Download from [ImageMagick](https://imagemagick.org/script/download.php#windows) and ensure you check "Install legacy utilities (e.g. convert)".
4.  **Groq API Key**: Get your free API key from [Groq Console](https://console.groq.com/).

## 📦 Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/lodonkontill-cyber/kliperr.git
    cd kliperr
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Create a `.env` file in the root directory and add your Groq API Key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## 🖥️ Usage

1.  Open `main.py` and configure the settings:
    - `YOUTUBE_URL`: The link to the video you want to process.
    - `JUMLAH_KLIP`: How many clips you want to generate.
    - `FONT_TYPE`: Ensure the font exists on your system (default is 'Arial-Bold').

2.  Run the script:
    ```bash
    python main.py
    ```

3.  Wait for the processing to finish. The output clips will be saved in the `hasil_shorts` folder.

## ⚙️ Configuration

You can customize the subtitle style and positioning in `main.py`:

```python
FONT_SIZE = 70
FONT_COLOR = '#FFD700' # Gold
FONT_COLOR_ALT = 'white'
POSISI_TEKS_Y = 0.75 # 75% height
```

## 📝 Troubleshooting

- **ImageMagick Error**: If MoviePy can't find ImageMagick, uncomment and update the path in `main.py`:
  ```python
  change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})
  ```
- **Whisper Device**: The script automatically detects CUDA (GPU) if available, otherwise it defaults to CPU.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📜 License

MIT License
