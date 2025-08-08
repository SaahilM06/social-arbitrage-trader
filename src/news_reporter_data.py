from googleapiclient.discovery import build
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable not set. Please add it to your .env file.")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


tickers = [
    'AAPL', 'MSFT', 'TSLA', 'GOOG', 'META', 'NVDA',
    'RBLX', 'AMC', 'GME', 'CROX',

    'ANF', 'URBN', 'AEO', 'LEVI', 'GPS', 'ASO', 'LULU', 'NKE',

    'JNJ', 'PFE', 'UNH', 'MRK', 'LLY', 'ABT'
]
output = {}

for ticker in tickers:
    query = f"{ticker} stock prediction"
    req = youtube.search().list(
        part="snippet",
        q=query,
        channelId="UCIALMKvObZNtJ6AmdCLP7Lg",  # Bloomberg Television
        type="video",
        order="date",
        publishedAfter="2025-04-01T00:00:00Z",
        maxResults=25
    )
    res = req.execute()
    # pull out only the fields you care about
    vids = [{
        "videoId": item["id"]["videoId"],
        "title":   item["snippet"]["title"],
        "publishedAt": item["snippet"]["publishedAt"]
    } for item in res["items"]]

    output[ticker] = vids

    # be nice to YouTube’s quota/servers
    time.sleep(1)

# write it all out
with open("bloomberg_video_catalog.json", "w") as f:
    json.dump(output, f, indent=2)

print("Done — results written to bloomberg_video_catalog.json")
