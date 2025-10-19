# ...existing code...
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import logging
import nltk
import logging

# Load API keys securely from Streamlit Secrets
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT = st.secrets.get("REDDIT_USER_AGENT", "GameLensBot/1.0")
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]

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

st.set_page_config(page_title="GameLens — AI Buzz & Stock Predictor", layout="wide", page_icon="🎮")

# New high-tech gamer CSS + animated background + font
st.markdown(
    """
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    /* Full animated gradient + subtle grid */
    :root{
      --bg1: #020617;
      --bg2: #071028;
      --accent1: #7b24ff;
      --accent2: #00ffc2;
      --accent3: #00a3ff;
      --muted: #9fb3c8;
      --card: rgba(6,10,16,0.6);
    }
    html, body, [class*="css"] {
        height:100%;
        background: radial-gradient(1200px 400px at 10% 10%, rgba(123,36,255,0.06), transparent 10%),
                    radial-gradient(800px 300px at 90% 90%, rgba(0,255,194,0.03), transparent 10%),
                    linear-gradient(120deg, var(--bg1) 0%, var(--bg2) 100%);
        color: #e6f0ff;
        font-family: 'Share Tech Mono', monospace;
        overflow: auto;
    }

    /* subtle animated noise / scanlines */
    .gamelens-wrap {
      min-height: 100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:48px 28px;
    }
    .gamelens-vc {
      width: 1200px;
      max-width: calc(100% - 48px);
      margin: auto;
      display:block;
    }

    /* header */
    .gl-header {
      font-family: 'Orbitron', sans-serif;
      text-align:center;
      margin-bottom:16px;
    }
    .gl-title {
      font-size:48px;
      font-weight:700;
      margin:0;
      letter-spacing:1px;
      background: linear-gradient(90deg, #7b24ff, #00ffc2, #00a3ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 6px 18px rgba(0,0,0,0.6));
      animation: hue 6s linear infinite;
    }
    @keyframes hue { 0%{filter: hue-rotate(0deg)} 50%{filter: hue-rotate(30deg)} 100%{filter: hue-rotate(0deg)} }

    .gl-sub {
      color: var(--accent2);
      opacity:0.95;
      margin-top:6px;
      font-size:14px;
    }

    /* neon card */
    .neon-card {
      background: var(--card);
      border-radius:12px;
      padding:16px;
      border: 1px solid rgba(123,36,255,0.08);
      box-shadow: 0 8px 40px rgba(0,0,0,0.6), 0 0 20px rgba(123,36,255,0.02) inset;
      margin-bottom:16px;
    }
    .neon-grid {
      display:grid;
      grid-template-columns: 1fr 420px;
      gap:18px;
      align-items:start;
    }

    /* input label style before actual Streamlit inputs */
    .input-label {
      font-family: 'Share Tech Mono', monospace;
      color: var(--muted);
      font-size:13px;
      margin-bottom:6px;
      display:flex;
      align-items:center;
      gap:8px;
    }
    .input-label .icon { font-size:18px; }

    /* buttons */
    .stButton>button {
      background: linear-gradient(90deg,var(--accent1), var(--accent2)) !important;
      color: #021014 !important;
      font-weight:700;
      border-radius:10px;
      padding:10px 20px;
      border: none;
      box-shadow: 0 14px 30px rgba(123,36,255,0.12);
      transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton>button:hover{
      transform: translateY(-3px);
      box-shadow: 0 20px 40px rgba(123,36,255,0.18), 0 0 30px rgba(0,255,194,0.06);
    }

    /* result / prediction */
    .result-panel {
      border-radius:10px;
      padding:14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
      border: 1px solid rgba(0,255,194,0.06);
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }
    .prediction {
      text-align:center;
      font-family: 'Orbitron', sans-serif;
      font-size:48px;
      margin:8px 0;
      letter-spacing:2px;
    }
    .pred-up {
      color:#00ff8c;
      text-shadow: 0 0 16px rgba(0,255,140,0.28), 0 0 6px rgba(0,160,120,0.2);
      animation: glowUp 2s ease-in-out infinite;
    }
    .pred-down {
      color:#ff4d6d;
      text-shadow: 0 0 16px rgba(255,77,109,0.28), 0 0 6px rgba(160,30,60,0.2);
      animation: glowDown 2s ease-in-out infinite;
    }
    @keyframes glowUp { 0%,100%{filter:brightness(1)} 50%{filter:brightness(1.18) saturate(1.05)} }
    @keyframes glowDown { 0%,100%{filter:brightness(1)} 50%{filter:brightness(1.18) saturate(1.05)} }

    /* footer */
    .gl-footer {
      text-align:center;
      color: #8fb9c9;
      font-size:13px;
      margin-top:20px;
      opacity:0.9;
    }

    /* responsive tweaks */
    @media (max-width:1100px) {
      .neon-grid { grid-template-columns: 1fr; }
      .gl-title { font-size:38px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# layout wrapper start
st.markdown('<div class="gamelens-wrap"><div class="gamelens-vc">', unsafe_allow_html=True)

# Header with styled title & subtitle
st.markdown(
    """
    <div class="gl-header">
      <div style="display:flex;align-items:center;justify-content:center;gap:12px;">
        <div style="width:64px;height:64px;border-radius:12px;background:linear-gradient(135deg,#7b24ff,#00ffc2);display:flex;align-items:center;justify-content:center;box-shadow: 0 10px 30px rgba(0,0,0,0.6);">
          <span style="font-size:34px;">🎮</span>
        </div>
      </div>
      <h1 class="gl-title">🎮 GameLens: AI Buzz & Stock Predictor</h1>
      <div class="gl-sub">Collect buzz across YouTube / Reddit / News / Trends — preprocess and predict company stock direction</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ensure VADER lexicon
nltk.download("vader_lexicon", quiet=True)
sid = SentimentIntensityAnalyzer()

# Model uploader & info (kept same logic, presented in neon card)
st.markdown('<div class="neon-grid">', unsafe_allow_html=True)

st.markdown('<div class="neon-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label"><span class="icon">🧠</span><strong>Model</strong></div>', unsafe_allow_html=True)
model_file = st.file_uploader("", type=["pkl","joblib"], key="model_file_uploader")
model = None
if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.success("Model loaded from upload.")
    except Exception as e:
        st.error(f"Failed to load uploaded model: {e}")
else:
    try:
        model = joblib.load("stock_direction_model.pkl")
        st.success("Loaded model from ./stock_direction_model.pkl")
    except Exception:
        st.info("No model loaded. Upload a model or place stock_direction_model.pkl in the app folder.")
st.markdown('<div style="margin-top:8px;color:#9fb3c8;font-size:12px">Model expects features: mentions, avg_sentiment, yt_views, reddit_comments</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Inputs card (aligned beside model uploader)
st.markdown('<div class="neon-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label"><span class="icon">🎮</span><strong>Game Name</strong></div>', unsafe_allow_html=True)
game_name = st.text_input("", value="", key="game_name").strip()
cols = st.columns([1,1,1])
with cols[0]:
    st.markdown('<div class="input-label"><span class="icon">⏳</span><strong>Days Back</strong></div>', unsafe_allow_html=True)
    days_back = st.number_input("", min_value=1, max_value=30, value=3, step=1, key="days_back")
with cols[1]:
    st.markdown('<div class="input-label"><span class="icon">💹</span><strong>Stock Ticker (optional)</strong></div>', unsafe_allow_html=True)
    override_ticker = st.text_input("", value="", key="override_ticker")
with cols[2]:
    st.markdown('<div style="height:46px;"></div>', unsafe_allow_html=True)
    st.markdown("")  # placeholder for alignment
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close neon-grid

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

# Main run (button on main page)
if st.button("Collect buzz & predict"):
    if not game_name:
        st.error("Enter a game name.")
    else:
        with st.spinner("Collecting data from APIs... this can take a few seconds"):
            ticker = detect_ticker(game_name)
            yt_videos = youtube_videos_and_stats(YOUTUBE_API_KEY, game_name, days_back)
reddit_posts = fetch_reddit(game_name, days_back, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)

            news_df = news_sentiments(game_name, days_back)
            trends_df = get_google_trends(game_name, days_back)
            feature_df = aggregate_features(game_name, days_back, yt_videos, reddit_posts, news_df, trends_df)
            st.success("Collection finished.")
        st.markdown('<div class="neon-card result-panel">', unsafe_allow_html=True)
        st.subheader("Collected daily features")
        st.dataframe(feature_df)
        # prepare input for model: use last available row
        if feature_df.empty:
            st.error("No features collected. Check API keys or try fewer days.")
        else:
            X = feature_df[["mentions","avg_sentiment","yt_views","reddit_comments"]].fillna(0)
            last = X.iloc[-1:].astype(float)
            st.markdown('<div style="margin-top:10px;"><div class="input-label"><span class="icon">🔎</span><strong>Features used (most recent)</strong></div></div>', unsafe_allow_html=True)
            st.write(last.T)
            if model is None:
                st.error("No model available. Upload or place stock_direction_model.pkl in folder.")
            else:
                try:
                    probs = model.predict_proba(last.values)[0]
                    classes = model.classes_
                    pred = int(model.predict(last.values)[0])
                    prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
                    # big glowing prediction
                    if pred == 1:
                        st.markdown(f'<div class="prediction pred-up">UP 📈</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction pred-down">DOWN/FLAT 📉</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align:center;color:var(--muted);margin-bottom:8px">Class probabilities: {prob_map}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align:center;color:var(--muted);margin-bottom:8px">Detected/used ticker: {detect_ticker(game_name) or "None / user override?"}</div>', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

# footer + closing wrapper
st.markdown('<div class="gl-footer">© 2025 GameLens — Built for gamers, powered by AI ⚡</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("Notes: Provide API keys where required in the code or upload a model. The app will still attempt prediction using available signals.")
st.markdown("Notes: This tester assumes the model expects features in the order [mentions, avg_sentiment, yt_views, reddit_comments]. If your model was trained with different preprocessing, ensure the uploaded CSV or manual inputs match the training pipeline.")

