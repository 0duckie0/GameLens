# ...existing code...
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import logging

# Optional imports (wrapped so app doesn't crash if user hasn't installed them)
try:
    from googleapiclient.discovery import build as youtube_build
except Exception:
    youtube_build = None
try:
    import praw
except Exception:
    praw = None
try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None
try:
    from gnews import GNews
except Exception:
    GNews = None
try:
    import finnhub
except Exception:
    finnhub = None

st.set_page_config(page_title="GameLens — Auto Buzz Collector & Stock Predictor", layout="centered")
st.title("GameLens — Give a game name, I'll collect buzz and predict stock direction")

# ensure VADER lexicon
nltk.download("vader_lexicon", quiet=True)
sid = SentimentIntensityAnalyzer()

# Sidebar: API keys + model upload
st.sidebar.header("API keys & model")
YOUTUBE_API_KEY = st.sidebar.text_input("YouTube Data API key", type="password")
REDDIT_CLIENT_ID = st.sidebar.text_input("Reddit client_id", type="password")
REDDIT_CLIENT_SECRET = st.sidebar.text_input("Reddit client_secret", type="password")
REDDIT_USER_AGENT = st.sidebar.text_input("Reddit user_agent", value="GameLensBot/0.1")

FINNHUB_API_KEY = 'd3pn8o1r01qmuiu9pon0d3pn8o1r01qmuiu9pong'

model_file = st.sidebar.file_uploader("Upload trained model (.pkl/.joblib) or leave empty to use ./stock_direction_model.pkl", type=["pkl","joblib"])
model = None
if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.sidebar.success("Model loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")
else:
    try:
        model = joblib.load("stock_direction_model.pkl")
        st.sidebar.success("Loaded model from ./stock_direction_model.pkl")
    except Exception:
        st.sidebar.info("No model loaded. Upload a model in sidebar or place stock_direction_model.pkl in the app folder.")

st.sidebar.markdown("---")
st.sidebar.write("Model expects features (case-sensitive): mentions, avg_sentiment, yt_views, reddit_comments")

# User inputs
st.subheader("Inputs")
game_name = st.text_input("Game name (example: 'Cyberpunk 2077')").strip()
days_back = st.number_input("Days to collect (past N days)", min_value=1, max_value=30, value=3, step=1)
override_ticker = st.text_input("Optional: override company ticker (leave empty to auto-detect)")

# minimal mapping (expandable)
GAME_TO_TICKER = {
     # ----- Take-Two Interactive -----
    "gta": "TTWO",
    "grand theft auto": "TTWO",
    "red dead": "TTWO",
    "nba 2k": "TTWO",
    "bully": "TTWO",

    # ----- Activision Blizzard -----
    "call of duty": "ATVI",
    "overwatch": "ATVI",
    "diablo": "ATVI",
    "warcraft": "ATVI",
    "starcraft": "ATVI",

    # ----- Electronic Arts -----
    "fifa": "EA",
    "ea sports fc": "EA",
    "battlefield": "EA",
    "apex legends": "EA",
    "sims": "EA",
    "mass effect": "EA",
    "dragon age": "EA",

    # ----- Ubisoft -----
    "assassin": "UBSFY",
    "far cry": "UBSFY",
    "rainbow six": "UBSFY",
    "ghost recon": "UBSFY",
    "watch dogs": "UBSFY",
    "just dance": "UBSFY",

    # ----- Sony Interactive Entertainment -----
    "god of war": "SONY",
    "spiderman": "SONY",
    "horizon": "SONY",
    "uncharted": "SONY",
    "last of us": "SONY",
    "ghost of tsushima": "SONY",
    "bloodborne": "SONY",

    # ----- Microsoft (Xbox, Bethesda, Mojang, etc.) -----
    "halo": "MSFT",
    "forza": "MSFT",
    "minecraft": "MSFT",
    "starfield": "MSFT",
    "elder scrolls": "MSFT",
    "fallout": "MSFT",
    "doom": "MSFT",
    "gears of war": "MSFT",
    "flight simulator": "MSFT",

    # ----- Nintendo -----
    "mario": "NTDOY",
    "zelda": "NTDOY",
    "pokemon": "NTDOY",
    "metroid": "NTDOY",
    "kirby": "NTDOY",
    "smash bros": "NTDOY",
    "splatoon": "NTDOY",

    # ----- Tencent (Riot Games, etc.) -----
    "valorant": "TCEHY",
    "league of legends": "TCEHY",
    "wild rift": "TCEHY",
    "arena of valor": "TCEHY",
    "pubg mobile": "TCEHY",

    # ----- Krafton (PUBG) -----
    "pubg": "259960.KQ",
    "battlegrounds": "259960.KQ",

    # ----- CD Projekt -----
    "cyberpunk": "OTGLY",
    "witcher": "OTGLY",

    # ----- Bandai Namco -----
    "elden ring": "NCBDF",
    "tekken": "NCBDF",
    "dark souls": "NCBDF",
    "dragon ball": "NCBDF",

    # ----- Capcom -----
    "resident evil": "CCOEY",
    "monster hunter": "CCOEY",
    "street fighter": "CCOEY",
    "devil may cry": "CCOEY",

    # ----- Square Enix -----
    "final fantasy": "SQNXF",
    "kingdom hearts": "SQNXF",
    "nier": "SQNXF",
    "tomb raider": "SQNXF",

    # ----- Sega Sammy -----
    "sonic": "SGAMY",
    "yakuza": "SGAMY",
    "persona": "SGAMY",

    # ----- Take-Two Private Division -----
    "kerbal": "TTWO",

    # ----- Epic Games (Private, no stock) -----
    "fortnite": "EPIC",
    "rocket league": "EPIC",
    "fall guys": "EPIC",

    # ----- Valve (Private, no stock) -----
    "counter strike": "VALVE",
    "dota": "VALVE",
    "half life": "VALVE",
    "portal": "VALVE",

    # ----- Embracer Group -----
    "dead island": "THQQF",
    "saints row": "THQQF",
    "remnant": "THQQF",
    "tomb raider (2023)": "THQQF",

    # ----- Roblox -----
    "roblox": "RBLX",

    # ----- Take-Two subsidiary (Zynga) -----
    "farmville": "TTWO",
    "words with friends": "TTWO",
}

# helpers
def detect_ticker(name):
    if override_ticker:
        return override_ticker.strip().upper()
    name_l = name.lower()
    for k, v in GAME_TO_TICKER.items():
        if k in name_l and v:
            return v
    return None

def sentiment_score(text):
    try:
        return sid.polarity_scores(str(text))["compound"]
    except Exception:
        return 0.0

# Google Trends
def get_google_trends(game, days):
    if TrendReq is None:
        return pd.DataFrame()
    try:
        pytrends = TrendReq(hl="en-US", tz=0)
        start_date = (datetime.utcnow().date() - timedelta(days=days-1)).strftime("%Y-%m-%d")
        timeframe = f"{start_date} {datetime.utcnow().date().strftime('%Y-%m-%d')}"
        pytrends.build_payload([game], timeframe=timeframe)
        data = pytrends.interest_over_time()
        if data.empty:
            return pd.DataFrame()
        df = data.reset_index()[["date", game]].rename(columns={game: "google_trends"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception as e:
        logging.debug("google trends error: %s", e)
        return pd.DataFrame()

# YouTube fetch
def youtube_videos_and_stats(api_key, query, days):
    if not api_key or youtube_build is None:
        return []
    try:
        youtube = youtube_build("youtube", "v3", developerKey=api_key)
        published_after = (datetime.utcnow().date() - timedelta(days=days-1)).isoformat() + "T00:00:00Z"
        published_before = (datetime.utcnow().date()).isoformat() + "T23:59:59Z"
        req = youtube.search().list(q=query, part="snippet", type="video", publishedAfter=published_after,
                                    publishedBefore=published_before, maxResults=25, order="relevance")
        res = req.execute()
        videos = []
        ids = []
        for it in res.get("items", []):
            vid = it["id"]["videoId"]
            ids.append(vid)
            videos.append({
                "videoId": vid,
                "title": it["snippet"]["title"],
                "description": it["snippet"].get("description",""),
                "publishedAt": it["snippet"]["publishedAt"]
            })
        if not ids:
            return videos
        stats_res = youtube.videos().list(part="statistics,contentDetails", id=",".join(ids)).execute()
        stats_map = {}
        for it in stats_res.get("items", []):
            sid_ = it["id"]
            s = it.get("statistics", {})
            stats_map[sid_] = {
                "viewCount": int(s.get("viewCount", 0)),
                "commentCount": int(s.get("commentCount", 0)) if s.get("commentCount") else 0
            }
        for v in videos:
            s = stats_map.get(v["videoId"], {})
            v.update(s)
        return videos
    except Exception as e:
        logging.debug("youtube error: %s", e)
        return []

# Reddit fetch
def fetch_reddit(query, days, client_id, client_secret, user_agent):
    if praw is None or not (client_id and client_secret and user_agent):
        return []
    try:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, check_for_async=False)
        subreddit = reddit.subreddit("all")
        results = []
        for submission in subreddit.search(query, limit=200, time_filter="week"):
            created = datetime.utcfromtimestamp(submission.created_utc).date()
            cutoff = datetime.utcnow().date() - timedelta(days=days-1)
            if created < cutoff:
                continue
            results.append({
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc
            })
        return results
    except Exception as e:
        logging.debug("reddit error: %s", e)
        return []

# News sentiment via GNews
def news_sentiments(query, days):
    if GNews is None:
        return pd.DataFrame()
    try:
        gn = GNews(language="en", country="US", period=f"{days}d", max_results=50)
        news = gn.get_news(query)
        records = []
        for n in news:
            text = (n.get("title","") + " " + n.get("description","")).strip()
            pub = n.get("published date")
            date_only = None
            try:
                if pub:
                    date_only = pd.to_datetime(pub, errors="coerce").date()
            except Exception:
                date_only = None
            records.append({"date": date_only, "text": text, "sent": sentiment_score(text)})
        return pd.DataFrame(records)
    except Exception as e:
        logging.debug("gnews error: %s", e)
        return pd.DataFrame()

# Aggregate daily features
def aggregate_features(game, days, yt_videos, reddit_posts, news_df, trends_df):
    days_range = pd.date_range(end=datetime.utcnow().date(), periods=days).date
    rows = []
    for d in days_range:
        # youtube
        yt_items = [v for v in yt_videos if pd.to_datetime(v.get("publishedAt", ""), errors="coerce").date() == d] if yt_videos else []
        yt_views = sum(v.get("viewCount",0) for v in yt_items)
        # reddit
        r_items = [r for r in reddit_posts if datetime.utcfromtimestamp(r["created_utc"]).date() == d] if reddit_posts else []
        reddit_comments = sum(r.get("num_comments",0) for r in r_items)
        # mentions & sentiment from news + titles
        texts = []
        texts += [v.get("title","") + " " + v.get("description","") for v in yt_items]
        texts += [r.get("title","") + " " + r.get("selftext","") for r in r_items]
        news_day = news_df[news_df["date"] == d] if not news_df.empty else pd.DataFrame()
        texts += news_day["text"].tolist() if not news_day.empty else []
        mentions = len([t for t in texts if t.strip()])
        sentiments = [sentiment_score(t) for t in texts] if texts else [0.0]
        avg_sent = float(np.mean(sentiments)) if sentiments else 0.0
        # google trends value
        gt_val = 0
        if not trends_df.empty:
            r = trends_df[trends_df["date"] == d]
            if not r.empty:
                gt_val = float(r["google_trends"].iloc[0])
        rows.append({
            "date": d,
            "mentions": mentions,
            "avg_sentiment": avg_sent,
            "yt_views": int(yt_views),
            "reddit_comments": int(reddit_comments),
            "google_trends": gt_val
        })
    return pd.DataFrame(rows)

# Main run
if st.button("Collect buzz & predict"):
    if not game_name:
        st.error("Enter a game name.")
    else:
        with st.spinner("Collecting data from APIs... this can take a few seconds"):
            ticker = detect_ticker(game_name)
            yt_videos = youtube_videos_and_stats('AIzaSyBgScLH0dr_6mCMAFbQmx241ASq8cPxyHM', game_name, days_back)
            reddit_posts = fetch_reddit(game_name, days_back, '2G4oG9HXTgQAAdd7jPZm6w', 'JTrGBRudnOf6iw27StNq764itkr5-A', 'Duckie')
            news_df = news_sentiments(game_name, days_back)
            trends_df = get_google_trends(game_name, days_back)
            feature_df = aggregate_features(game_name, days_back, yt_videos, reddit_posts, news_df, trends_df)
            st.success("Collection finished.")
        st.subheader("Collected daily features")
        st.dataframe(feature_df)
        # prepare input for model: use last available row
        if feature_df.empty:
            st.error("No features collected. Check API keys or try fewer days.")
        else:
            X = feature_df[["mentions","avg_sentiment","yt_views","reddit_comments"]].fillna(0)
            last = X.iloc[-1:].astype(float)
            st.write("Features used for prediction (most recent):")
            st.write(last.T)
            if model is None:
                st.error("No model available. Upload or place stock_direction_model.pkl in folder.")
            else:
                try:
                    probs = model.predict_proba(last.values)[0]
                    classes = model.classes_
                    pred = int(model.predict(last.values)[0])
                    prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
                    st.success(f"Predicted direction: {'UP 📈' if pred == 1 else 'DOWN/FLAT 📉'} (class {pred})")
                    st.write("Class probabilities:", prob_map)
                    st.write("Detected/used ticker:", detect_ticker(game_name) or "None / user override?")
                    # optional: if finnhub key provided, show recent company sentiment (best-effort)
                    if FINNHUB_API_KEY and finnhub is not None and detect_ticker(game_name):
                        try:
                            client = finnhub.Client(api_key= FINNHUB_API_KEY)
                            comp = client.general_news('general', min_id=0)
                            st.write("Fetched company news (sample):", (comp[:5]))
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("Notes: Provide API keys in the sidebar. If some APIs are missing the app will still attempt prediction using available signals. Ensure the uploaded model expects features: mentions, avg_sentiment, yt_views, reddit_comments.")
# ...existing code...
st.markdown("Notes: This tester assumes the model expects features in the order [mentions, avg_sentiment, yt_views, reddit_comments]. If your model was trained with different preprocessing, ensure the uploaded CSV or manual inputs match the training pipeline.")