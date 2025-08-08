import os
import yt_dlp

VIDEOS = {
    "AAPL": [
        "https://www.youtube.com/watch?v=th8QQ_NJpss",
        "https://www.youtube.com/watch?v=faTIEPk7RRs",
    ],
    "TSLA": [
        "https://www.youtube.com/watch?v=sJAu7ByJje8",
        "https://www.youtube.com/watch?v=b5n96lKrwjU",
        "https://www.youtube.com/watch?v=yLIidCOy94I",
        "https://www.youtube.com/watch?v=CuO0nySpc_Q",
    ],
    "GPS": [
        "https://www.youtube.com/watch?v=N29MhzMYpyM",
    ],
    "UNH": [
        "https://www.youtube.com/watch?v=WMGsUyZXCc0",
    ],
}

for ticker, urls in VIDEOS.items():
    folder = f"videos/{ticker}"
    os.makedirs(folder, exist_ok=True)

    for url in urls:
        video_id = url.split("v=")[-1]
        output_path = f"{folder}/{video_id}.mp4"

        opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": output_path,
        }

        print(f"Downloading {url} → {output_path}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
