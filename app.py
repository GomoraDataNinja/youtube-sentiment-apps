import streamlit as st
import pandas as pd
import re
from googleapiclient.discovery import build
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="YouTube Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --primary-blue: #1a73e8;
    --secondary-blue: #185abc;
    --accent-teal: #34a853;
    --success-green: #188038;
    --warning-orange: #f29900;
    --danger-red: #d93025;
    --light-bg: #f1f3f4;
    --card-bg: #ffffff;
    --text-dark: #202124;
    --text-muted: #5f6368;
}

/* App background */
.stApp {
    background: #f1f3f4;
    font-family: "Inter", "Roboto", "Segoe UI", system-ui, sans-serif;
    color: var(--text-dark);
}

/* Main sections */
.section {
    background: var(--card-bg);
    padding: 26px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(60,64,67,.15);
    border: 1px solid #dadce0;
}

/* Headings */
h1 {
    color: var(--text-dark);
    font-weight: 600;
    font-size: 2.2rem;
}

h2, h3 {
    color: var(--text-dark);
    font-weight: 500;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dadce0;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: var(--primary-blue);
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 500;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--secondary-blue);
}

/* Metrics */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(60,64,67,.15);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 8px;
    padding: 12px 22px;
    border: 1px solid #dadce0;
    color: var(--text-muted);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: var(--primary-blue);
    color: white !important;
    border: none;
}

/* Comment cards */
.comment-card {
    background: white;
    border-radius: 10px;
    border: 1px solid #dadce0;
}

/* Reduce visual noise */
.stPlotlyChart {
    background: white;
    border-radius: 12px;
    padding: 10px;
}

/* Tables */
.stDataFrame {
    background: white;
    border-radius: 10px;
}

/* Footer */
footer {
    color: var(--text-muted);
}
</style>

""", unsafe_allow_html=True)

# Configuration
API_KEY = st.secrets["youtube_api_key"]

# Initialize session state for multiple videos
if 'video_data' not in st.session_state:
    st.session_state.video_data = {}
if 'current_videos' not in st.session_state:
    st.session_state.current_videos = []

# Custom Plotly template
custom_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(font=dict(size=20, color='#212529')),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', title_font=dict(size=14)),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)', title_font=dict(size=14)),
        legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.1)'),
        colorway=['#4361ee', '#7209b7', '#4cc9f0', '#38b000', '#f48c06', '#f72585']
    )
)

# Utility Functions
def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_comments(video_id, video_url):
    """Fetch comments from YouTube API"""
    if video_id in st.session_state.video_data:
        return st.session_state.video_data[video_id]
    
    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    try:
        # First get video details
        video_request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        video_response = video_request.execute()
        
        if not video_response['items']:
            return None
            
        video_info = video_response['items'][0]
        
        # Fetch comments
        all_comments = []
        next_page_token = None
        
        for _ in range(2):  # Get up to 200 comments (2 pages)
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append({
                    "comment": snippet["textDisplay"],
                    "published_at": snippet["publishedAt"],
                    "like_count": snippet.get("likeCount", 0),
                    "author": snippet.get("authorDisplayName", "Unknown")
                })
            
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        
        if not all_comments:
            return None
            
        df = pd.DataFrame(all_comments)
        df["published_at"] = pd.to_datetime(df["published_at"])
        
        # Store in session state
        st.session_state.video_data[video_id] = {
            "df": df,
            "title": video_info['snippet']['title'],
            "url": video_url,
            "stats": video_info['statistics']
        }
        
        return st.session_state.video_data[video_id]
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def analyze_sentiment(df):
    """Analyze sentiment using TextBlob"""
    if df.empty:
        return df
    
    df["sentiment_score"] = df["comment"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    
    df["sentiment"] = df["sentiment_score"].apply(
        lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral"
    )
    
    return df

def generate_insights(df, video_title):
    """Generate key insights from sentiment analysis"""
    total_comments = len(df)
    if total_comments == 0:
        return ["No comments to analyze"]
    
    sentiment_counts = df["sentiment"].value_counts()
    avg_sentiment = df["sentiment_score"].mean()
    
    insights = []
    
    # Overall sentiment insight
    if avg_sentiment > 0.2:
        insights.append(f"💚 **Very Positive Reception**: Comments show strong positive sentiment (avg: {avg_sentiment:.2f})")
    elif avg_sentiment > 0:
        insights.append(f"👍 **Generally Positive**: Overall feedback is positive (avg: {avg_sentiment:.2f})")
    elif avg_sentiment < -0.2:
        insights.append(f"🔴 **Strong Criticism**: Significant negative feedback detected (avg: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        insights.append(f"⚠️ **Mixed with Concerns**: Some negative feedback present (avg: {avg_sentiment:.2f})")
    else:
        insights.append(f"⚖️ **Neutral Dominance**: Comments are mostly neutral or balanced")
    
    # Distribution insight
    dominant_sentiment = sentiment_counts.idxmax()
    dominant_percent = (sentiment_counts.max() / total_comments) * 100
    insights.append(f"📊 **{dominant_sentiment} Comments Dominate**: {dominant_percent:.1f}% of all comments")
    
    # Top negative keywords insight
    if "Negative" in sentiment_counts:
        negative_comments = df[df["sentiment"] == "Negative"]["comment"]
        if len(negative_comments) > 0:
            negative_text = " ".join(negative_comments.str.lower())
            words = [word for word in negative_text.split() if len(word) > 3]
            common_words = Counter(words).most_common(3)
            if common_words:
                word_list = ", ".join([word for word, _ in common_words])
                insights.append(f"🔍 **Common Concerns**: Frequent words in negative comments: {word_list}")
    
    # Engagement insight
    if 'like_count' in df.columns:
        avg_likes = df["like_count"].mean()
        if avg_likes > 10:
            insights.append(f"🔥 **High Engagement**: Comments average {avg_likes:.0f} likes each")
        elif avg_likes > 5:
            insights.append(f"👏 **Good Engagement**: Comments receive decent likes ({avg_likes:.0f} avg)")
    
    return insights

def create_comparison_chart(video_ids):
    """Create comparison chart for multiple videos"""
    if len(video_ids) < 2:
        return None
    
    comparison_data = []
    
    for vid in video_ids:
        if vid in st.session_state.video_data:
            data = st.session_state.video_data[vid]
            # Ensure sentiment analysis is performed
            df = analyze_sentiment(data["df"].copy())
            
            # Calculate metrics safely
            total_comments = len(df)
            if total_comments == 0:
                # Handle empty comment case
                comparison_data.append({
                    "Video": data.get("title", "Unknown Video")[:30] + "...",
                    "Total Comments": 0,
                    "Positive %": 0,
                    "Negative %": 0,
                    "Avg Sentiment": 0,
                    "Video ID": vid
                })
                continue
            
            positive_pct = (df["sentiment"] == "Positive").mean() * 100
            negative_pct = (df["sentiment"] == "Negative").mean() * 100
            avg_sentiment = df["sentiment_score"].mean()
            
            comparison_data.append({
                "Video": data.get("title", "Unknown Video")[:30] + "...",
                "Total Comments": total_comments,
                "Positive %": positive_pct,
                "Negative %": negative_pct,
                "Avg Sentiment": avg_sentiment,
                "Video ID": vid
            })
    
    if comparison_data:
        return pd.DataFrame(comparison_data)
    
    return None

def to_csv_bytes(df):
    """Convert DataFrame to CSV bytes"""
    return df.to_csv(index=False).encode("utf-8")

# Sidebar
st.sidebar.title("🎯 YouTube Sentiment Analyzer")
st.sidebar.markdown("---")

# Video input section
st.sidebar.subheader("📹 Add Videos to Analyze")
video_url = st.sidebar.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Add Video", use_container_width=True):
        if video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                if video_id not in st.session_state.current_videos:
                    # Enhanced loading spinner
                    with st.spinner(''):
                        st.markdown("""
                        <div style="text-align: center; padding: 40px;">
                            <div style="
                                width: 60px;
                                height: 60px;
                                border: 4px solid #f3f3f3;
                                border-top: 4px solid #4361ee;
                                border-radius: 50%;
                                animation: spin 1s linear infinite;
                                margin: 0 auto;
                            "></div>
                            <p style="margin-top: 20px; color: #6c757d;">Analyzing comments...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        result = fetch_comments(video_id, video_url)
                        if result:
                            st.session_state.current_videos.append(video_id)
                            st.success("Video added successfully!")
                        else:
                            st.error("Failed to fetch video data")
                else:
                    st.warning("Video already added")
            else:
                st.error("Invalid YouTube URL")
        else:
            st.error("Please enter a URL")

with col2:
    if st.button("Clear All", use_container_width=True, type="secondary"):
        st.session_state.current_videos = []
        st.rerun()

st.sidebar.markdown("---")

# Display current videos
if st.session_state.current_videos:
    st.sidebar.subheader("📋 Selected Videos")
    for i, vid in enumerate(st.session_state.current_videos, 1):
        video_info = st.session_state.video_data.get(vid, {})
        title = video_info.get("title", f"Video {i}")
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.caption(f"{i}. {title[:40]}...")
        with col2:
            if st.button("✕", key=f"remove_{vid}"):
                st.session_state.current_videos.remove(vid)
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Add multiple videos for comparison")

# Main Content
st.title("📊 YouTube Comment Sentiment Dashboard")

if not st.session_state.current_videos:
    st.info("👈 Add YouTube videos from the sidebar to get started!")
    st.markdown("""
    <div class="section">
    <h3>How to use:</h3>
    <ol>
        <li>Paste a YouTube video URL in the sidebar</li>
        <li>Click "Add Video" to analyze comments</li>
        <li>Add multiple videos for comparison</li>
        <li>Use tabs below to explore insights</li>
    </ol>
    <p><strong>Note:</strong> You need a valid YouTube Data API key</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Single Video Analysis",
    "⚖️ Compare Videos",
    "📋 Comment Explorer",
    "📊 Advanced Analytics"
])

# Tab 1: Single Video Analysis
with tab1:
    if st.session_state.current_videos:
        video_selector = st.selectbox(
            "Select Video to Analyze",
            options=st.session_state.current_videos,
            format_func=lambda x: st.session_state.video_data[x]["title"][:50] + "..."
        )
        
        if video_selector:
            video_info = st.session_state.video_data[video_selector]
            df = analyze_sentiment(video_info["df"].copy())
            
            # Enhanced Insight Boxes
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("🔍 Key Insights")
            
            insight_colors = {
                '💚': 'linear-gradient(135deg, #38b000, #2d8c00)',
                '👍': 'linear-gradient(135deg, #4cc9f0, #4361ee)',
                '🔴': 'linear-gradient(135deg, #d00000, #9d0208)',
                '⚠️': 'linear-gradient(135deg, #f48c06, #e85d04)',
                '⚖️': 'linear-gradient(135deg, #7209b7, #560bad)',
                '📊': 'linear-gradient(135deg, #4361ee, #3a0ca3)',
                '🔍': 'linear-gradient(135deg, #f72585, #b5179e)',
                '🔥': 'linear-gradient(135deg, #ff5400, #ff6d00)',
                '👏': 'linear-gradient(135deg, #ff9e00, #ff9100)'
            }
            
            insights = generate_insights(df, video_info["title"])
            
            for insight in insights:
                # Extract emoji for color matching
                emoji = insight[:2] if insight[:2] in insight_colors else '📌'
                bg_color = insight_colors.get(emoji, 'linear-gradient(135deg, #6c757d, #495057)')
                
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    color: white;
                    padding: 20px;
                    border-radius: 14px;
                    margin: 12px 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    border-left: 6px solid rgba(255,255,255,0.3);
                ">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.8rem;">{emoji}</span>
                        <span style="font-size: 1rem; line-height: 1.4;">{insight}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced Metrics Section
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📊 Performance Overview")
            
            positive_pct = (df["sentiment"] == "Positive").mean() * 100
            engagement = df["like_count"].mean() if "like_count" in df.columns else 0
            
            # Create metric cards with icons
            metrics_html = f"""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0;">
                <div class="metric-card" style="background: linear-gradient(135deg, #4cc9f0, #4361ee);">
                    <div style="font-size: 2rem;">📝</div>
                    <div style="font-size: 2rem; font-weight: bold;">{len(df)}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">Total Comments</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #38b000, #2d8c00);">
                    <div style="font-size: 2rem;">📈</div>
                    <div style="font-size: 2rem; font-weight: bold;">{df['sentiment_score'].mean():.2f}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">Avg Sentiment</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #7209b7, #560bad);">
                    <div style="font-size: 2rem;">👍</div>
                    <div style="font-size: 2rem; font-weight: bold;">{positive_pct:.1f}%</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">Positive</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #f48c06, #dc6b06);">
                    <div style="font-size: 2rem;">❤️</div>
                    <div style="font-size: 2rem; font-weight: bold;">{engagement:.0f}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">Avg Likes</div>
                </div>
            </div>
            
            <style>
            .metric-card {{
                padding: 25px;
                border-radius: 14px;
                color: white;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }}
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            </style>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="section">', unsafe_allow_html=True)
                st.subheader("📊 Sentiment Distribution")
                
                sentiment_counts = df["sentiment"].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#38b000',
                        'Neutral': '#6c757d',
                        'Negative': '#d00000'
                    }
                )
                fig.update_layout(
                    template=custom_template,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="section">', unsafe_allow_html=True)
                st.subheader("📈 Sentiment Over Time")
                
                daily_avg = df.groupby(df["published_at"].dt.date)["sentiment_score"].mean()
                daily_count = df.groupby(df["published_at"].dt.date).size()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_avg.index,
                    y=daily_avg.values,
                    mode='lines+markers',
                    name='Avg Sentiment',
                    line=dict(color='#4361ee', width=3),
                    marker=dict(size=8, color='#4361ee')
                ))
                fig.add_trace(go.Bar(
                    x=daily_count.index,
                    y=daily_count.values,
                    name='Comment Volume',
                    yaxis='y2',
                    marker_color='rgba(67, 97, 238, 0.2)',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    yaxis=dict(title="Sentiment Score", gridcolor='rgba(0,0,0,0.05)'),
                    yaxis2=dict(
                        title="Comment Count",
                        overlaying="y",
                        side="right",
                        gridcolor='rgba(0,0,0,0.05)'
                    ),
                    hovermode='x unified',
                    template=custom_template,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Word Analysis
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("🔍 Word Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Words in Positive Comments**")
                positive_text = " ".join(df[df["sentiment"] == "Positive"]["comment"].str.lower())
                words = [word for word in positive_text.split() if len(word) > 3 and word.isalpha()]
                if words:
                    word_counts = Counter(words).most_common(10)
                    pos_words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                    # Style the dataframe
                    st.dataframe(
                        pos_words_df.style
                        .background_gradient(subset=['Count'], cmap='Greens')
                        .format({'Count': '{:,.0f}'}),
                        use_container_width=True,
                        height=400
                    )
            
            with col2:
                st.markdown("**Top Words in Negative Comments**")
                negative_text = " ".join(df[df["sentiment"] == "Negative"]["comment"].str.lower())
                words = [word for word in negative_text.split() if len(word) > 3 and word.isalpha()]
                if words:
                    word_counts = Counter(words).most_common(10)
                    neg_words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                    # Style the dataframe
                    st.dataframe(
                        neg_words_df.style
                        .background_gradient(subset=['Count'], cmap='Reds')
                        .format({'Count': '{:,.0f}'}),
                        use_container_width=True,
                        height=400
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Compare Videos
with tab2:
    if len(st.session_state.current_videos) >= 2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("📊 Video Comparison")
        
        comparison_df = create_comparison_chart(st.session_state.current_videos)
        
        if comparison_df is not None and not comparison_df.empty:
            # Comparison Metrics
            cols = st.columns(len(comparison_df))
            for idx, (_, row) in enumerate(comparison_df.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #4361ee, #3a0ca3);
                        color: white;
                        padding: 20px;
                        border-radius: 14px;
                        text-align: center;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 10px;">
                            {row["Video"]}
                        </div>
                        <div style="font-size: 2rem; font-weight: 700;">
                            {row["Total Comments"]}
                        </div>
                        <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 5px;">
                            Comments
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9rem;">
                            Sentiment: {row['Avg Sentiment']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Comparison Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    comparison_df,
                    x="Video",
                    y=["Positive %", "Negative %"],
                    title="Sentiment Distribution by Video",
                    barmode="group",
                    color_discrete_map={"Positive %": "#38b000", "Negative %": "#d00000"}
                )
                fig.update_layout(
                    template=custom_template,
                    yaxis_title="Percentage (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    comparison_df,
                    x="Total Comments",
                    y="Avg Sentiment",
                    size="Total Comments",
                    color="Video",
                    title="Engagement vs Sentiment",
                    hover_name="Video",
                    size_max=60,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_layout(
                    template=custom_template,
                    xaxis_title="Total Comments",
                    yaxis_title="Average Sentiment",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Comparison Table
            st.subheader("📋 Detailed Comparison")
            display_df = comparison_df.drop(columns=["Video ID"])
            st.dataframe(
                display_df.style
                .background_gradient(subset=["Positive %"], cmap="Greens", vmin=0, vmax=100)
                .background_gradient(subset=["Negative %"], cmap="Reds_r", vmin=0, vmax=100)
                .background_gradient(subset=["Avg Sentiment"], cmap="RdYlGn", vmin=-1, vmax=1)
                .format({
                    "Positive %": "{:.1f}%",
                    "Negative %": "{:.1f}%",
                    "Avg Sentiment": "{:.3f}",
                    "Total Comments": "{:,.0f}"
                })
                .set_properties(**{
                    'text-align': 'center',
                    'font-family': 'Inter, sans-serif'
                }),
                use_container_width=True,
                height=300
            )
        else:
            st.info("Unable to generate comparison data. Make sure videos have comments.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Add at least 2 videos to enable comparison")

# Tab 3: Comment Explorer
with tab3:
    if st.session_state.current_videos:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("💬 Comment Explorer")
        
        selected_video = st.selectbox(
            "Select Video",
            options=st.session_state.current_videos,
            format_func=lambda x: st.session_state.video_data[x]["title"][:50] + "...",
            key="explorer_select"
        )
        
        if selected_video:
            video_info = st.session_state.video_data[selected_video]
            df = analyze_sentiment(video_info["df"].copy())
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment",
                    options=["Positive", "Neutral", "Negative"],
                    default=["Positive", "Neutral", "Negative"],
                    format_func=lambda x: f"📊 {x}"
                )
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Newest", "Oldest", "Most Likes", "Highest Sentiment", "Lowest Sentiment"],
                    format_func=lambda x: f"🔽 {x}" if "Lowest" in x else f"🔼 {x}" if "Highest" in x else f"📅 {x}"
                )
            with col3:
                comments_to_show = st.slider("Comments to show", 10, 100, 20, format="%d comments")
            
            # Apply filters
            filtered_df = df[df["sentiment"].isin(sentiment_filter)]
            
            # Apply sorting
            sort_mapping = {
                "Newest": ("published_at", False),
                "Oldest": ("published_at", True),
                "Most Likes": ("like_count", False),
                "Highest Sentiment": ("sentiment_score", False),
                "Lowest Sentiment": ("sentiment_score", True)
            }
            
            if sort_by in sort_mapping:
                col, ascending = sort_mapping[sort_by]
                filtered_df = filtered_df.sort_values(col, ascending=ascending)
            
            # Display comments
            st.subheader(f"📝 Showing {len(filtered_df.head(comments_to_show))} comments")
            
            sentiment_colors = {
                "Positive": "#38b000",
                "Neutral": "#6c757d",
                "Negative": "#d00000"
            }
            
            sentiment_icons = {
                "Positive": "✅",
                "Neutral": "⚪",
                "Negative": "❌"
            }
            
            for _, row in filtered_df.head(comments_to_show).iterrows():
                sentiment_color = sentiment_colors[row["sentiment"]]
                sentiment_icon = sentiment_icons[row["sentiment"]]
                
                st.markdown(f"""
                <div style="
                    border-left: 6px solid {sentiment_color};
                    padding: 18px;
                    margin: 12px 0;
                    background: white;
                    border-radius: 0 12px 12px 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    transition: all 0.2s ease;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.5rem;">{sentiment_icon}</span>
                            <span style="font-weight: 600; color: {sentiment_color}; font-size: 1.1rem;">
                                {row['sentiment']} <span style="font-weight: 400; opacity: 0.8;">({row['sentiment_score']:.2f})</span>
                            </span>
                        </div>
                        <div style="color: #6c757d; font-size: 0.9em; text-align: right;">
                            <div>{row['published_at'].strftime('%Y-%m-%d %H:%M')}</div>
                            <div style="margin-top: 4px;">
                                {row.get('author', 'Unknown')}
                                {' • ' + str(row['like_count']) + ' ❤️' if row.get('like_count', 0) > 0 else ''}
                            </div>
                        </div>
                    </div>
                    <div style="
                        padding: 12px;
                        background: #f8f9fa;
                        border-radius: 8px;
                        margin-top: 8px;
                        font-size: 0.95rem;
                        line-height: 1.5;
                        color: #495057;
                    ">
                        {row['comment'][:400]}{'...' if len(row['comment']) > 400 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Advanced Analytics
with tab4:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📈 Advanced Analytics")
    
    if len(st.session_state.current_videos) >= 1:
        selected_video = st.selectbox(
            "Select Video for Analysis",
            options=st.session_state.current_videos,
            format_func=lambda x: st.session_state.video_data[x]["title"][:50] + "...",
            key="advanced_select"
        )
        
        if selected_video:
            video_info = st.session_state.video_data[selected_video]
            df = analyze_sentiment(video_info["df"].copy())
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment Distribution by Hour
                st.markdown("**🕒 Sentiment by Hour of Day**")
                df["hour"] = df["published_at"].dt.hour
                hourly_sentiment = df.groupby("hour")["sentiment_score"].mean().reset_index()
                
                fig = px.bar(
                    hourly_sentiment,
                    x="hour",
                    y="sentiment_score",
                    title="Average Sentiment by Hour",
                    color="sentiment_score",
                    color_continuous_scale="RdYlGn",
                    labels={"hour": "Hour of Day", "sentiment_score": "Average Sentiment"}
                )
                fig.update_layout(
                    template=custom_template,
                    xaxis=dict(tickmode='linear', dtick=1),
                    coloraxis_colorbar=dict(title="Sentiment")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Comment Length vs Sentiment
                st.markdown("**📏 Comment Length Analysis**")
                df["comment_length"] = df["comment"].str.len()
                
                fig = px.scatter(
                    df,
                    x="comment_length",
                    y="sentiment_score",
                    color="sentiment",
                    title="Comment Length vs Sentiment",
                    hover_data=["comment"],
                    color_discrete_map={
                        "Positive": "#38b000",
                        "Neutral": "#6c757d",
                        "Negative": "#d00000"
                    },
                    labels={"comment_length": "Comment Length (characters)", "sentiment_score": "Sentiment Score"}
                )
                fig.update_layout(
                    template=custom_template,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export Section
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📤 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download CSV", use_container_width=True):
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Click to download",
                        data=csv,
                        file_name=f"youtube_sentiment_{selected_video}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📊 Generate Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        # Generate summary statistics
                        summary = {
                            "Video Title": video_info["title"],
                            "Total Comments": len(df),
                            "Average Sentiment": df["sentiment_score"].mean(),
                            "Positive Comments": (df["sentiment"] == "Positive").sum(),
                            "Negative Comments": (df["sentiment"] == "Negative").sum(),
                            "Neutral Comments": (df["sentiment"] == "Neutral").sum(),
                            "Max Comment Likes": df["like_count"].max() if "like_count" in df.columns else 0,
                            "Average Comment Length": df["comment"].str.len().mean(),
                            "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        summary_df = pd.DataFrame([summary]).T
                        summary_df.columns = ["Value"]
                        
                        st.markdown("### 📋 Analysis Summary")
                        st.dataframe(
                            summary_df.style
                            .background_gradient(subset=['Value'], cmap='Blues')
                            .format({
                                "Average Sentiment": "{:.3f}",
                                "Average Comment Length": "{:.1f}",
                                "Total Comments": "{:,.0f}",
                                "Positive Comments": "{:,.0f}",
                                "Negative Comments": "{:,.0f}",
                                "Neutral Comments": "{:,.0f}",
                                "Max Comment Likes": "{:,.0f}"
                            }),
                            use_container_width=True,
                            height=400
                        )
                        
                        st.success("✅ Report generated successfully!")
            
            with col3:
                if st.button("📈 View Raw Data", use_container_width=True):
                    st.markdown("### 📄 Raw Comment Data")
                    st.dataframe(
                        df[["comment", "sentiment", "sentiment_score", "published_at", "like_count", "author"]]
                        .style
                        .background_gradient(subset=['sentiment_score'], cmap='RdYlGn', vmin=-1, vmax=1)
                        .format({
                            'sentiment_score': '{:.3f}',
                            'like_count': '{:,.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px 0;">
    <div style="font-size: 1rem; font-weight: 600; margin-bottom: 10px;">
        YouTube Sentiment Analysis Dashboard
    </div>
    <div style="font-size: 0.9rem;">
        Powered by YouTube Data API v3 • Built with Streamlit
    </div>
    <div style="margin-top: 10px; font-size: 0.8rem; opacity: 0.7;">
        Analyze and visualize sentiment in YouTube comments
    </div>
</div>
""", unsafe_allow_html=True)