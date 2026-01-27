import streamlit as st
import pandas as pd
import re
from googleapiclient.discovery import build
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import time
import os
import hashlib
import warnings

warnings.filterwarnings("ignore")

APP_VERSION = "2.2.1"
APP_NAME = "YouTube Sentiment Analysis"
DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "production")
SESSION_TIMEOUT_MINUTES = 60
MAX_COMMENTS_PER_VIDEO = 500

st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def get_org_password():
    env_pw = os.environ.get("APP_PASSWORD", "").strip()
    if env_pw:
        return env_pw
    try:
        sec_pw = str(st.secrets.get("app_password", "")).strip()
        if sec_pw:
            return sec_pw
    except Exception:
        pass
    return "youtube2024"

ORG_PASSWORD = get_org_password()

THEME = {
    "bg": "#ffffff",
    "panel": "#ffffff",
    "panel2": "#f7f7f7",
    "text": "#111111",
    "muted": "#5b5b5b",
    "border": "rgba(0,0,0,0.10)",
    "border2": "rgba(0,0,0,0.14)",
    "accent": "#d71e28",
    "accent2": "#b5161f",
    "good": "#168a45",
    "bad": "#d11a2a",
    "neutral": "#6b7280",
}

SENTIMENT_COLORS = {
    "Positive": THEME["good"],
    "Neutral": THEME["neutral"],
    "Negative": THEME["bad"],
}

def apply_style(mode="app"):
    login_mode = (mode == "login")
    block_display = "flex" if login_mode else "block"
    block_justify = "center" if login_mode else "initial"
    block_align = "center" if login_mode else "initial"
    block_min_h = "100vh" if login_mode else "auto"
    block_pt = "0rem" if login_mode else "1.6rem"
    block_pb = "0rem" if login_mode else "2.2rem"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {THEME['bg']};
            --panel: {THEME['panel']};
            --panel2: {THEME['panel2']};
            --text: {THEME['text']};
            --muted: {THEME['muted']};
            --border: {THEME['border']};
            --border2: {THEME['border2']};
            --accent: {THEME['accent']};
            --accent2: {THEME['accent2']};
            --good: {THEME['good']};
            --bad: {THEME['bad']};
            --neutral: {THEME['neutral']};
        }}

        .stApp {{
            background: var(--bg);
        }}

        /* Lock container spacing so deploy renders the same */
        [data-testid="stAppViewContainer"] {{
            background: var(--bg) !important;
        }}

        .block-container {{
            max-width: 1080px !important;
            padding-top: {block_pt} !important;
            padding-bottom: {block_pb} !important;

            display: {block_display} !important;
            justify-content: {block_justify} !important;
            align-items: {block_align} !important;
            min-height: {block_min_h} !important;
        }}

        html, body, [class*="css"] {{
            color: var(--text) !important;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif !important;
        }}

        h1, h2, h3, h4, h5, h6,
        p, span, label, small, li,
        div, section, header, footer,
        .stMarkdown, .stCaption, .stText, .stAlert,
        [data-testid="stMarkdownContainer"] * {{
            color: var(--text) !important;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif !important;
        }}

        .muted, .stCaption {{
            color: var(--muted) !important;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}

        section[data-testid="stSidebar"] {{
            background: #ffffff !important;
            border-right: 1px solid var(--border) !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--text) !important;
        }}

        .card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 18px;
        }}

        .card-soft {{
            background: var(--panel2);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 18px;
        }}

        .hero {{
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 26px 22px;
            background:
                radial-gradient(900px 260px at 50% -10%, rgba(215,30,40,0.10), transparent 60%),
                linear-gradient(180deg, #ffffff, #ffffff);
        }}

        .title {{
            font-size: 28px;
            font-weight: 800;
            letter-spacing: 0.2px;
            margin: 0;
        }}

        .subtitle {{
            margin-top: 6px;
            color: var(--muted) !important;
            font-size: 14px;
            line-height: 1.6;
        }}

        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #ffffff;
            font-size: 12px;
            font-weight: 650;
            color: var(--muted) !important;
        }}

        .chip-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            display: inline-block;
            background: var(--accent);
        }}

        .metric {{
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px 14px;
            background: #ffffff;
        }}

        .metric-k {{
            font-size: 12px;
            color: var(--muted) !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.9px;
        }}

        .metric-v {{
            font-size: 26px;
            font-weight: 850;
            margin-top: 6px;
        }}

        /* Buttons */
        div.stButton > button {{
            background: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 14px !important;
            padding: 0.7rem 1rem !important;
            font-weight: 750 !important;
            color: #ffffff !important;
        }}

        div.stButton > button:hover {{
            background: var(--accent2) !important;
            border: 1px solid var(--accent2) !important;
        }}

        /* Inputs */
        .stTextInput > div > div > input {{
            background-color: #ffffff !important;
            border: 1px solid var(--border2) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
        }}

        .stSelectbox > div > div {{
            background-color: #ffffff !important;
            border: 1px solid var(--border2) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab"] {{
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            margin-right: 8px !important;
            padding: 10px 12px !important;
            font-weight: 750 !important;
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background: rgba(215,30,40,0.10) !important;
            border: 1px solid rgba(215,30,40,0.35) !important;
        }}

        /* Dataframe */
        [data-testid="stDataFrame"] {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }}
        [data-testid="stDataFrame"] * {{
            color: var(--text) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def touch():
    st.session_state.last_activity = datetime.now()

def is_timed_out():
    last = st.session_state.get("last_activity")
    if not last:
        return False
    return (datetime.now() - last).total_seconds() > SESSION_TIMEOUT_MINUTES * 60

def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    safe_rerun()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
if "video_data" not in st.session_state:
    st.session_state.video_data = {}
if "current_videos" not in st.session_state:
    st.session_state.current_videos = []
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()

def login_screen():
    apply_style(mode="login")

    st.markdown(
        f"""
        <div style="width: 440px; max-width: 92vw;">
            <div class="card">
                <div class="title">{APP_NAME}</div>
                <div class="subtitle">
                    Secure. Fast. Easy.<br>
                    Enter the organisation password to continue.
                </div>
                <div style="height: 14px;"></div>
                <div class="chip"><span class="chip-dot"></span> Version {APP_VERSION} • {DEPLOYMENT_MODE.title()}</div>
            </div>
            <div style="height: 12px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=True):
        pw = st.text_input("Password", type="password", placeholder="Organisation password")
        ok = st.form_submit_button("Sign in", use_container_width=True)

    if ok:
        if pw == ORG_PASSWORD:
            st.session_state.authenticated = True
            touch()
            safe_rerun()
        else:
            st.error("Wrong password.")

if st.session_state.authenticated and is_timed_out():
    st.session_state.authenticated = False
    st.warning("Session timed out. Sign in again.")
    login_screen()
    st.stop()

if not st.session_state.authenticated:
    login_screen()
    st.stop()

apply_style(mode="app")
touch()

def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{{11}})",
        r"youtu\.be\/([0-9A-Za-z_-]{{11}})",
        r"embed\/([0-9A-Za-z_-]{{11}})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

def youtube_client():
    try:
        api_key = st.secrets["youtube_api_key"]
    except Exception:
        api_key = None

    if not api_key or not str(api_key).strip():
        st.error("Missing YouTube API key. Add youtube_api_key in st.secrets.")
        return None

    return build("youtube", "v3", developerKey=api_key)

def get_video_comments(youtube, video_id, max_comments=MAX_COMMENTS_PER_VIDEO):
    all_comments = []
    next_page_token = None

    while len(all_comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()

            for item in response.get("items", []):
                s = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append(
                    {
                        "comment": s.get("textDisplay", ""),
                        "published_at": s.get("publishedAt", ""),
                        "like_count": s.get("likeCount", 0),
                        "author": s.get("authorDisplayName", "Unknown"),
                    }
                )

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        except Exception as e:
            msg = str(e)
            if "commentsDisabled" in msg:
                return []
            return []

    return all_comments

def fetch_video(video_id, video_url):
    if video_id in st.session_state.video_data:
        return st.session_state.video_data[video_id]

    yt = youtube_client()
    if yt is None:
        return None

    try:
        vr = yt.videos().list(part="snippet,statistics,status", id=video_id).execute()
        if not vr.get("items"):
            st.error("Video not found or not accessible.")
            return None

        info = vr["items"][0]
        comments = get_video_comments(yt, video_id, max_comments=MAX_COMMENTS_PER_VIDEO)

        if not comments:
            st.error("No comments returned. Comments may be disabled for this video.")
            return None

        df = pd.DataFrame(comments)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        payload = {
            "df": df,
            "title": info.get("snippet", {}).get("title", "Unknown Title"),
            "url": video_url,
            "stats": info.get("statistics", {}),
        }

        st.session_state.video_data[video_id] = payload
        return payload

    except Exception as e:
        msg = str(e)
        if "commentsDisabled" in msg:
            st.error("This video has comments disabled. Pick another video.")
            return None
        if "quotaExceeded" in msg:
            st.error("YouTube API quota exceeded. Try again later or use a new API key/project.")
            return None
        if "keyInvalid" in msg or "API key not valid" in msg:
            st.error("Your YouTube API key is invalid. Check st.secrets['youtube_api_key'].")
            return None
        if "accessNotConfigured" in msg:
            st.error("YouTube Data API v3 is not enabled for this Google Cloud project.")
            return None
        if "forbidden" in msg or "403" in msg:
            st.error("Request forbidden (403). Video may be restricted/private or comments unavailable.")
            return None

        st.error(f"Error fetching data: {msg}")
        return None

def analyze_sentiment(df):
    if df is None or df.empty:
        return df

    out = df.copy()
    out["sentiment_score"] = out["comment"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    out["sentiment"] = out["sentiment_score"].apply(
        lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral"
    )
    return out

def donut_chart(sentiment_counts, title, center_text):
    labels = list(sentiment_counts.index)
    values = list(sentiment_counts.values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=[SENTIMENT_COLORS.get(x, THEME["neutral"]) for x in labels]),
                textinfo="percent",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16, color=THEME["text"])),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text"]),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="left", x=0),
        annotations=[
            dict(
                text=center_text,
                x=0.5,
                y=0.5,
                font=dict(size=14, color=THEME["text"]),
                showarrow=False,
            )
        ],
    )
    return fig

def build_comparison(video_ids):
    rows = []
    for vid in video_ids:
        data = st.session_state.video_data.get(vid)
        if not data:
            continue
        df = analyze_sentiment(data["df"])
        if df is None or df.empty:
            continue

        rows.append(
            {
                "Video": data["title"][:55],
                "Total Comments": len(df),
                "Positive %": (df["sentiment"] == "Positive").mean() * 100,
                "Neutral %": (df["sentiment"] == "Neutral").mean() * 100,
                "Negative %": (df["sentiment"] == "Negative").mean() * 100,
                "Avg Sentiment": df["sentiment_score"].mean(),
            }
        )

    return pd.DataFrame(rows) if rows else None

def sentiment_badge(s):
    c = SENTIMENT_COLORS.get(s, THEME["neutral"])
    return f"""
    <span style="
        display:inline-flex;
        align-items:center;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid {THEME['border']};
        background: rgba(0,0,0,0.02);
        font-weight: 700;
        font-size: 12px;
        gap: 8px;
    ">
        <span style="width:8px;height:8px;border-radius:999px;background:{c};display:inline-block;"></span>
        {s}
    </span>
    """

# Top bar
top1, top2, top3 = st.columns([3, 1.4, 1])
with top1:
    st.markdown(
        f"""
        <div class="hero">
            <div class="title">{APP_NAME}</div>
            <div class="subtitle">Paste a YouTube link, then review sentiment and export results.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top2:
    st.markdown(
        f"""
        <div class="card" style="height: 100%;">
            <div class="chip"><span class="chip-dot"></span> Secure session</div>
            <div style="height: 10px;"></div>
            <div class="chip">Session {st.session_state.session_id}</div>
            <div style="height: 10px;"></div>
            <div class="chip">Mode {DEPLOYMENT_MODE.title()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top3:
    if st.button("Logout", use_container_width=True):
        logout()

st.markdown("")

# Add video area (simple)
a1, a2 = st.columns([3, 1])
with a1:
    video_url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")

with a2:
    if st.button("Add video", use_container_width=True):
        url = (video_url or "").strip()
        if not url:
            st.error("Paste a YouTube URL.")
        else:
            vid = extract_video_id(url)
            if not vid:
                st.error("That URL does not look like a valid YouTube video link.")
            else:
                if vid in st.session_state.current_videos:
                    st.warning("That video is already added.")
                else:
                    with st.spinner("Fetching video data..."):
                        data = fetch_video(vid, url)
                    if data:
                        st.session_state.current_videos.append(vid)
                        st.success("Video added.")
                        safe_rerun()

# Selected videos (compact)
if st.session_state.current_videos:
    with st.expander("Selected videos", expanded=False):
        for vid in st.session_state.current_videos:
            data = st.session_state.video_data.get(vid, {})
            name = data.get("title", vid)
            r1, r2 = st.columns([5, 1])
            with r1:
                st.write(name)
            with r2:
                if st.button("Remove", key=f"rm_{vid}", use_container_width=True):
                    st.session_state.current_videos.remove(vid)
                    if vid in st.session_state.video_data:
                        del st.session_state.video_data[vid]
                    safe_rerun()

cbtn1, _ = st.columns([1, 5])
with cbtn1:
    if st.session_state.current_videos:
        if st.button("Clear all", use_container_width=True):
            st.session_state.current_videos = []
            st.session_state.video_data = {}
            safe_rerun()

st.markdown("")

if not st.session_state.current_videos:
    st.markdown(
        """
        <div class="card-soft">
            <div style="font-size:16px; font-weight:800;">Secure. Fast. Easy.</div>
            <div class="subtitle">
                Add one video to view insights.<br>
                Add two or more videos to compare.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Global metrics
total_videos = len(st.session_state.current_videos)
total_comments = 0
scores = []

for vid in st.session_state.current_videos:
    data = st.session_state.video_data.get(vid)
    if not data:
        continue
    total_comments += len(data["df"])
    df0 = analyze_sentiment(data["df"])
    if df0 is not None and not df0.empty:
        scores.extend(df0["sentiment_score"].tolist())

avg_sent = float(np.mean(scores)) if scores else 0.0

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"<div class='metric'><div class='metric-k'>Videos</div><div class='metric-v'>{total_videos}</div></div>",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"<div class='metric'><div class='metric-k'>Comments</div><div class='metric-v'>{total_comments:,}</div></div>",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"<div class='metric'><div class='metric-k'>Avg sentiment</div><div class='metric-v'>{avg_sent:.2f}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("")

tabs = st.tabs(["Overview", "Explore", "Compare", "Export"])

with tabs[0]:
    vid = st.selectbox(
        "Video",
        options=st.session_state.current_videos,
        format_func=lambda x: st.session_state.video_data.get(x, {}).get("title", x)[:70],
        key="ov_vid",
    )
    data = st.session_state.video_data.get(vid)
    if not data:
        st.info("Video data missing. Re-add the video.")
        st.stop()

    df = analyze_sentiment(data["df"])
    if df is None or df.empty:
        st.info("No comments to analyze.")
        st.stop()

    s_counts = df["sentiment"].value_counts()
    dom = s_counts.idxmax()
    dom_pct = (s_counts.max() / len(df)) * 100
    center = f"{dom}<br>{dom_pct:.0f}%"

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:16px; font-weight:800;">Sentiment breakdown</div>
                <div class="subtitle">{data['title']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = donut_chart(s_counts, "Sentiment distribution", center)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="card">
                <div style="font-size:16px; font-weight:800;">Recent comments</div>
                <div class="subtitle">Latest feedback, limited so the page stays clean.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        recent = df.sort_values("published_at", ascending=False).head(8)
        for _, row in recent.iterrows():
            txt = str(row.get("comment", "")).strip()
            if len(txt) > 180:
                txt = txt[:180] + "..."
            when = row.get("published_at")
            when_txt = ""
            if pd.notna(when):
                when_txt = pd.to_datetime(when).strftime("%Y-%m-%d %H:%M")

            badge = sentiment_badge(row["sentiment"])
            likes = int(row.get("like_count", 0) or 0)
            like_txt = f" • {likes} likes" if likes > 0 else ""

            st.markdown(
                f"""
                <div class="card-soft" style="margin-top: 10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                        <div>{badge}</div>
                        <div class="muted" style="font-size:12px;">{when_txt}</div>
                    </div>
                    <div style="margin-top: 10px; font-size: 13px; line-height:1.55;">{txt}</div>
                    <div class="muted" style="margin-top: 8px; font-size:12px;">
                        {row.get('author','Unknown')}{like_txt}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tabs[1]:
    vid = st.selectbox(
        "Video",
        options=st.session_state.current_videos,
        format_func=lambda x: st.session_state.video_data.get(x, {}).get("title", x)[:70],
        key="ex_vid",
    )
    data = st.session_state.video_data.get(vid)
    if not data:
        st.info("Video data missing. Re-add the video.")
        st.stop()

    df = analyze_sentiment(data["df"])
    if df is None or df.empty:
        st.info("No comments to analyze.")
        st.stop()

    f1, f2, f3 = st.columns([1.3, 1.3, 1])
    with f1:
        sentiment_filter = st.multiselect(
            "Filter sentiment",
            options=["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
        )
    with f2:
        sort_by = st.selectbox(
            "Sort by",
            options=["Newest", "Oldest", "Most Likes", "Highest Sentiment", "Lowest Sentiment"],
        )
    with f3:
        n_show = st.slider("Rows", 10, 80, 20)

    filtered = df[df["sentiment"].isin(sentiment_filter)]

    mapping = {
        "Newest": ("published_at", False),
        "Oldest": ("published_at", True),
        "Most Likes": ("like_count", False),
        "Highest Sentiment": ("sentiment_score", False),
        "Lowest Sentiment": ("sentiment_score", True),
    }
    col, asc = mapping[sort_by]
    filtered = filtered.sort_values(col, ascending=asc)

    show = filtered.head(n_show).copy()
    show["published_at"] = show["published_at"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        show[["published_at", "author", "like_count", "sentiment", "sentiment_score", "comment"]],
        use_container_width=True,
        height=520,
    )

with tabs[2]:
    if len(st.session_state.current_videos) < 2:
        st.info("Add at least 2 videos to compare.")
        st.stop()

    comp = build_comparison(st.session_state.current_videos)
    if comp is None or comp.empty:
        st.info("No comparison data yet.")
        st.stop()

    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Video comparison</div>
            <div class="subtitle">Compare sentiment and engagement across selected videos.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    fig = px.bar(
        comp,
        x="Video",
        y=["Positive %", "Neutral %", "Negative %"],
        barmode="group",
        title="Sentiment by video",
        color_discrete_map={
            "Positive %": SENTIMENT_COLORS["Positive"],
            "Neutral %": SENTIMENT_COLORS["Neutral"],
            "Negative %": SENTIMENT_COLORS["Negative"],
        },
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text"]),
        title_font=dict(color=THEME["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comp.style.format(
            {
                "Total Comments": "{:,.0f}",
                "Positive %": "{:.1f}%",
                "Neutral %": "{:.1f}%",
                "Negative %": "{:.1f}%",
                "Avg Sentiment": "{:.3f}",
            }
        ),
        use_container_width=True,
        height=320,
    )

with tabs[3]:
    vid = st.selectbox(
        "Video",
        options=st.session_state.current_videos,
        format_func=lambda x: st.session_state.video_data.get(x, {}).get("title", x)[:70],
        key="xp_vid",
    )
    data = st.session_state.video_data.get(vid)
    if not data:
        st.info("Video data missing. Re-add the video.")
        st.stop()

    df = analyze_sentiment(data["df"])
    if df is None or df.empty:
        st.info("No comments to export.")
        st.stop()

    counts = df["sentiment"].value_counts()
    summary = pd.DataFrame(
        {
            "Video Title": [data["title"]],
            "Video ID": [vid],
            "Total Comments": [len(df)],
            "Avg Sentiment": [df["sentiment_score"].mean()],
            "Positive %": [(df["sentiment"] == "Positive").mean() * 100],
            "Neutral %": [(df["sentiment"] == "Neutral").mean() * 100],
            "Negative %": [(df["sentiment"] == "Negative").mean() * 100],
            "Positive Count": [int(counts.get("Positive", 0))],
            "Neutral Count": [int(counts.get("Neutral", 0))],
            "Negative Count": [int(counts.get("Negative", 0))],
            "Generated At": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        }
    )

    detailed = df.copy()
    detailed["video_id"] = vid
    detailed["video_title"] = data["title"]
    detailed["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("<div class='card'><div style='font-size:16px;font-weight:800;'>Summary export</div><div class='subtitle'>One row per video.</div></div>", unsafe_allow_html=True)
        st.download_button(
            "Download summary CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"youtube_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.markdown("<div class='card'><div style='font-size:16px;font-weight:800;'>Detailed export</div><div class='subtitle'>One row per comment.</div></div>", unsafe_allow_html=True)
        st.download_button(
            "Download detailed CSV",
            data=detailed.to_csv(index=False).encode("utf-8"),
            file_name=f"youtube_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("")
st.markdown(
    f"""
    <div class="card-soft" style="text-align:center;">
        <div style="font-weight:800;">{APP_NAME} v{APP_VERSION}</div>
        <div class="subtitle">Secure session • {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


