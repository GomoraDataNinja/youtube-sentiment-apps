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
.stApp {
    background-color: #f1f3f4;
    font-family: "Segoe UI", sans-serif;
}

.section {
    background-color: #ffffff;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
}

h1 {
    color: #202124;
}

h2, h3 {
    color: #3c4043;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f8f9fa;
    padding: 8px;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 12px 24px;
}

.stTabs [aria-selected="true"] {
    background-color: #1a73e8;
    color: white !important;
}

div[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

button {
    background-color: #1a73e8;
    color: white;
    border-radius: 6px;
}

.insight-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin: 15px 0;
}

.video-comparison {
    border-left: 4px solid #1a73e8;
    padding-left: 15px;
    margin: 10px 0;
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
                    with st.spinner("Fetching video data..."):
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
            
            # Key Insights Section
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("🔑 Key Insights")
            
            insights = generate_insights(df, video_info["title"])
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Comments", len(df))
            with col2:
                st.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")
            with col3:
                positive_pct = (df["sentiment"] == "Positive").mean() * 100
                st.metric("Positive %", f"{positive_pct:.1f}%")
            with col4:
                engagement = df["like_count"].mean() if "like_count" in df.columns else 0
                st.metric("Avg Likes/Comment", f"{engagement:.0f}")
            
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
                        'Positive': '#28a745',
                        'Neutral': '#6c757d',
                        'Negative': '#dc3545'
                    }
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
                    line=dict(color='#1a73e8', width=3)
                ))
                fig.add_trace(go.Bar(
                    x=daily_count.index,
                    y=daily_count.values,
                    name='Comment Volume',
                    yaxis='y2',
                    marker_color='rgba(200, 200, 200, 0.6)'
                ))
                
                fig.update_layout(
                    yaxis=dict(title="Sentiment Score"),
                    yaxis2=dict(
                        title="Comment Count",
                        overlaying="y",
                        side="right"
                    ),
                    hovermode='x unified'
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
                    st.dataframe(pos_words_df, use_container_width=True)
            
            with col2:
                st.markdown("**Top Words in Negative Comments**")
                negative_text = " ".join(df[df["sentiment"] == "Negative"]["comment"].str.lower())
                words = [word for word in negative_text.split() if len(word) > 3 and word.isalpha()]
                if words:
                    word_counts = Counter(words).most_common(10)
                    neg_words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                    st.dataframe(neg_words_df, use_container_width=True)
            
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
                    st.metric(
                        label=row["Video"],
                        value=row["Total Comments"],
                        delta=f"{row['Avg Sentiment']:.2f} avg sentiment"
                    )
            
            # Comparison Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    comparison_df,
                    x="Video",
                    y=["Positive %", "Negative %"],
                    title="Sentiment Distribution by Video",
                    barmode="group",
                    color_discrete_map={"Positive %": "#28a745", "Negative %": "#dc3545"}
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
                    size_max=60
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Comparison Table
            st.subheader("📋 Detailed Comparison")
            display_df = comparison_df.drop(columns=["Video ID"])
            st.dataframe(
                display_df.style
                .background_gradient(subset=["Positive %"], cmap="Greens")
                .background_gradient(subset=["Negative %"], cmap="Reds")
                .format({
                    "Positive %": "{:.1f}%",
                    "Negative %": "{:.1f}%",
                    "Avg Sentiment": "{:.2f}"
                }),
                use_container_width=True
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
                    default=["Positive", "Neutral", "Negative"]
                )
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Newest", "Oldest", "Most Likes", "Highest Sentiment", "Lowest Sentiment"]
                )
            with col3:
                comments_to_show = st.slider("Comments to show", 10, 100, 20)
            
            # Apply filters
            filtered_df = df[df["sentiment"].isin(sentiment_filter)]
            
            # Apply sorting
            if sort_by == "Newest":
                filtered_df = filtered_df.sort_values("published_at", ascending=False)
            elif sort_by == "Oldest":
                filtered_df = filtered_df.sort_values("published_at", ascending=True)
            elif sort_by == "Most Likes":
                filtered_df = filtered_df.sort_values("like_count", ascending=False)
            elif sort_by == "Highest Sentiment":
                filtered_df = filtered_df.sort_values("sentiment_score", ascending=False)
            elif sort_by == "Lowest Sentiment":
                filtered_df = filtered_df.sort_values("sentiment_score", ascending=True)
            
            # Display comments
            st.subheader(f"Showing {len(filtered_df.head(comments_to_show))} comments")
            
            for _, row in filtered_df.head(comments_to_show).iterrows():
                sentiment_color = {
                    "Positive": "#28a745",
                    "Neutral": "#6c757d",
                    "Negative": "#dc3545"
                }[row["sentiment"]]
                
                st.markdown(f"""
                <div style="
                    border-left: 4px solid {sentiment_color};
                    padding: 12px;
                    margin: 8px 0;
                    background: #f8f9fa;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: bold; color: {sentiment_color};">
                            {row['sentiment']} ({row['sentiment_score']:.2f})
                        </span>
                        <span style="color: #6c757d; font-size: 0.9em;">
                            {row['published_at'].strftime('%Y-%m-%d')}
                            {' • ' + str(row['like_count']) + ' likes' if row.get('like_count', 0) > 0 else ''}
                        </span>
                    </div>
                    <p style="margin: 8px 0;">{row['comment'][:300]}{'...' if len(row['comment']) > 300 else ''}</p>
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
                    color_continuous_scale="RdYlGn"
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
                        "Positive": "#28a745",
                        "Neutral": "#6c757d",
                        "Negative": "#dc3545"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export Section
            st.markdown("---")
            st.subheader("📤 Export Data")
            
            col1, col2 = st.columns(2)
            
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
                            "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        summary_df = pd.DataFrame([summary])
                        st.dataframe(summary_df, use_container_width=True)
                        
                        st.success("Report generated successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("YouTube Sentiment Analysis Dashboard • Powered by YouTube Data API v3")