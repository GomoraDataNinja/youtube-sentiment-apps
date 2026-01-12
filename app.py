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
import io
from fpdf import FPDF
import base64

st.set_page_config(
    page_title="YouTube Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Instructions / How-to section */
.instructions {
    background-color: #ffffff !important;
    color: #111111 !important;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
}

/* Force text inside */
.instructions h1,
.instructions h2,
.instructions h3,
.instructions p,
.instructions li {
    color: #111111 !important;
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* FORCE solid sidebar — no blur, no transparency */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    background-image: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border-right: 1px solid #dadce0;
}

/* Sidebar content wrapper */
section[data-testid="stSidebar"] > div {
    background-color: #ffffff !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #202124;
    font-family: "Inter", "Roboto", system-ui, sans-serif;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #202124;
    font-weight: 600;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #1a73e8;
    color: #ffffff;
    border-radius: 8px;
    border: none;
    font-weight: 500;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #185abc;
}

/* Sidebar text inputs */
section[data-testid="stSidebar"] input {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 6px;
    color: #202124;
}

/* Remove Streamlit blur overlay */
div[data-testid="stSidebarNav"] {
    background: #ffffff !important;
    backdrop-filter: none !important;
}

/* Sidebar tabs / captions */
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 8px;
}

/* Kill any remaining transparency */
section[data-testid="stSidebar"] {
    opacity: 1 !important;
}

/* Custom styles for metrics and cards */
.metric-card {
    padding: 20px;
    border-radius: 10px;
    background: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

.section {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border: 1px solid #f0f0f0;
}

/* FIX FOR INSTRUCTION SECTION - REMOVE BLUR/TRANSPARENCY */
div.element-container:has(div.stInfo) {
    background-color: white !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

div.stInfo {
    background-color: white !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* Target the specific div that contains the instruction markdown */
div.element-container:has(div[data-testid="stMarkdownContainer"]):has(p:contains("How to use:")) {
    background-color: white !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* Ensure the section div inside has solid background */
div.section {
    background-color: white !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# ENHANCED SANITIZATION FUNCTION
# ============================================
def sanitize_text_for_pdf(text, method='remove'):
    """
    Enhanced sanitization for PDF generation
    Returns ASCII-only text safe for FPDF
    """
    if text is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # If empty string, return as is
    if text.strip() == "":
        return text
    
    # First, replace problematic Unicode characters with ASCII equivalents
    replacements = {
        # Bullets and special symbols
        '\u2022': '*',      # Bullet -> asterisk
        '\u25cf': '*',      # Black circle -> asterisk
        '\u25e6': '*',      # White circle -> asterisk
        '\u2023': '*',      # Triangular bullet -> asterisk
        '\u2043': '-',      # Hyphen bullet -> hyphen
        
        # Dashes
        '\u2013': '-',      # En dash
        '\u2014': '--',     # Em dash
        
        # Quotes
        '\u2018': "'",      # Left single quotation
        '\u2019': "'",      # Right single quotation
        '\u201c': '"',      # Left double quotation
        '\u201d': '"',      # Right double quotation
        
        # Special symbols
        '\u00a9': '(c)',    # Copyright
        '\u00ae': '(R)',    # Registered
        '\u2122': '(TM)',   # Trademark
        '\u2026': '...',    # Ellipsis
        
        # Currency
        '\u00a3': 'GBP',    # Pound
        '\u20ac': 'EUR',    # Euro
        '\u00a5': 'JPY',    # Yen
        
        # Fractions
        '\u00bc': '1/4',    # 1/4
        '\u00bd': '1/2',    # 1/2
        '\u00be': '3/4',    # 3/4
        
        # Common emoji replacements
        '\u2764': '<3',     # Heart
        '\u2665': '<3',     # Heart suit
        '\u2605': '*',      # Star
        '\u2713': '[OK]',   # Check mark
        '\u2714': '[OK]',   # Heavy check
        '\u2717': '[X]',    # X mark
        
        # Arrows
        '\u2190': '<-',     # Left arrow
        '\u2192': '->',     # Right arrow
        '\u2191': '^',      # Up arrow
        '\u2193': 'v',      # Down arrow
    }
    
    # Apply replacements
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    # Handle remaining Unicode characters
    if method == 'remove':
        # Remove all non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
    elif method == 'replace':
        # Replace non-ASCII with placeholder
        result = []
        for char in text:
            if ord(char) < 128:
                result.append(char)
            else:
                result.append(' ')
        text = ''.join(result)
    
    # Clean up multiple spaces and trim
    text = ' '.join(text.split())
    
    return text

# ============================================
# ROBUST PDF GENERATION FUNCTION
# ============================================
def generate_pdf_report(video_id, video_info, df):
    """Generate a simple PDF report using built-in fonts only"""
    try:
        # Analyze data
        df_analyzed = analyze_sentiment(df.copy())
        total_comments = len(df_analyzed)
        
        # Create PDF with built-in fonts
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "YouTube Sentiment Analysis Report", 0, 1, "C")
        pdf.ln(5)
        
        # Horizontal line
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
        # Video info
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Video Information", 0, 1)
        pdf.set_font("Arial", "", 10)
        
        video_title = video_info.get("title", "Unknown Video")
        safe_title = sanitize_text_for_pdf(video_title, method='remove')[:80]
        
        pdf.cell(40, 6, "Title:", 0, 0)
        pdf.multi_cell(0, 6, safe_title)
        
        pdf.cell(40, 6, "Video ID:", 0, 0)
        pdf.cell(0, 6, video_id, 0, 1)
        
        pdf.cell(40, 6, "Analysis Date:", 0, 0)
        pdf.cell(0, 6, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 1)
        
        pdf.cell(40, 6, "Total Comments:", 0, 0)
        pdf.cell(0, 6, f"{total_comments:,}", 0, 1)
        pdf.ln(5)
        
        if total_comments > 0:
            # Calculate metrics
            avg_sentiment = df_analyzed["sentiment_score"].mean()
            positive_count = (df_analyzed["sentiment"] == "Positive").sum()
            negative_count = (df_analyzed["sentiment"] == "Negative").sum()
            neutral_count = (df_analyzed["sentiment"] == "Neutral").sum()
            
            positive_pct = (positive_count / total_comments) * 100
            negative_pct = (negative_count / total_comments) * 100
            neutral_pct = (neutral_count / total_comments) * 100
            
            # Key Metrics
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Key Metrics", 0, 1)
            pdf.set_font("Arial", "", 10)
            
            col_width = 95
            row_height = 8
            
            metrics = [
                ("Total Comments", f"{total_comments:,}"),
                ("Average Sentiment", f"{avg_sentiment:.3f}"),
                ("Positive Comments", f"{positive_count:,} ({positive_pct:.1f}%)"),
                ("Negative Comments", f"{negative_count:,} ({negative_pct:.1f}%)"),
                ("Neutral Comments", f"{neutral_count:,} ({neutral_pct:.1f}%)")
            ]
            
            for label, value in metrics:
                pdf.cell(col_width, row_height, label, 0, 0)
                pdf.cell(0, row_height, value, 0, 1)
                pdf.ln(2)
            
            pdf.ln(5)
            
            # Insights
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Key Insights", 0, 1)
            pdf.set_font("Arial", "", 10)
            
            insights = generate_insights(df_analyzed, video_info["title"])
            for i, insight in enumerate(insights[:5], 1):
                safe_insight = sanitize_text_for_pdf(insight, method='remove')
                pdf.multi_cell(0, 6, f"{i}. {safe_insight}")
                pdf.ln(2)
            
            # Sample Comments
            if total_comments > 0:
                pdf.add_page()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Sample Comments by Sentiment", 0, 1)
                pdf.ln(5)
                
                sentiments = ["Positive", "Neutral", "Negative"]
                
                for sentiment in sentiments:
                    pdf.set_font("Arial", "B", 11)
                    if sentiment == "Positive":
                        pdf.set_text_color(56, 176, 0)  # Green
                    elif sentiment == "Negative":
                        pdf.set_text_color(220, 53, 69)  # Red
                    else:
                        pdf.set_text_color(100, 100, 100)  # Gray
                    
                    pdf.cell(0, 8, f"{sentiment} Comments:", 0, 1)
                    pdf.set_text_color(0, 0, 0)  # Back to black
                    pdf.set_font("Arial", "", 9)
                    
                    sentiment_comments = df_analyzed[df_analyzed["sentiment"] == sentiment].head(3)
                    
                    if len(sentiment_comments) > 0:
                        for idx, (_, row) in enumerate(sentiment_comments.iterrows(), 1):
                            comment_text = row["comment"]
                            safe_comment = sanitize_text_for_pdf(comment_text, method='remove')
                            truncated_comment = safe_comment[:100] + "..." if len(safe_comment) > 100 else safe_comment
                            pdf.multi_cell(0, 5, f"  {idx}. {truncated_comment}")
                            pdf.ln(1)
                    
                    pdf.ln(5)
                    
                    if pdf.get_y() > 250:
                        pdf.add_page()
            
            # Recommendations
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 10, "Recommendations", 0, 1)
            pdf.set_font("Arial", "", 10)
            
            if avg_sentiment > 0.2:
                recs = [
                    "Leverage positive feedback in marketing",
                    "Engage with commenters to build community",
                    "Create similar content based on positive response"
                ]
            elif avg_sentiment > 0:
                recs = [
                    "Acknowledge positive feedback",
                    "Address minor concerns mentioned",
                    "Monitor sentiment trends"
                ]
            elif avg_sentiment < -0.2:
                recs = [
                    "Review critical feedback carefully",
                    "Consider content adjustments",
                    "Address common concerns publicly"
                ]
            else:
                recs = [
                    "Seek more specific feedback",
                    "Encourage viewer engagement",
                    "Test different content approaches"
                ]
            
            for rec in recs:
                pdf.multi_cell(0, 6, f"* {rec}")
                pdf.ln(2)
        else:
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, "No comments available for analysis.", 0, 1)
        
        # Footer
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, "Generated by YouTube Sentiment Dashboard", 0, 0, "C")
        
        # Return PDF as bytes using BytesIO
        buffer = io.BytesIO()
        
        # Get PDF output as string
        pdf_output = pdf.output(dest='S')
        
        # Encode to latin-1 with error replacement
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1', 'replace')
        else:
            pdf_bytes = pdf_output
        
        buffer.write(pdf_bytes)
        buffer.seek(0)
        
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        # Create a minimal fallback PDF
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "YouTube Sentiment Report", 0, 1, "C")
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, f"Video: {video_id}", 0, 1)
            pdf.cell(0, 10, f"Total Comments: {len(df)}", 0, 1)
            pdf.cell(0, 10, "Report generated with limited data", 0, 1)
            
            buffer = io.BytesIO()
            pdf_output = pdf.output(dest='S')
            if isinstance(pdf_output, str):
                pdf_bytes = pdf_output.encode('latin-1', 'replace')
            else:
                pdf_bytes = pdf_output
            buffer.write(pdf_bytes)
            buffer.seek(0)
            return buffer.getvalue()
        except:
            # Ultimate fallback
            return b""

# ============================================
# TEXT REPORT FALLBACK
# ============================================
def create_text_report_fallback(video_id, video_info, df):
    """Create a simple text-based report as fallback"""
    try:
        df_analyzed = analyze_sentiment(df.copy())
        total_comments = len(df_analyzed)
        
        report_text = f"""
{'=' * 60}
YouTube Sentiment Analysis Report
{'=' * 60}

VIDEO INFORMATION
{'=' * 60}
Title: {video_info.get('title', 'Unknown Video')}
Video ID: {video_id}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Comments: {total_comments:,}

"""
        
        if total_comments > 0:
            avg_sentiment = df_analyzed["sentiment_score"].mean()
            positive_pct = (df_analyzed["sentiment"] == "Positive").mean() * 100
            negative_pct = (df_analyzed["sentiment"] == "Negative").mean() * 100
            neutral_pct = (df_analyzed["sentiment"] == "Neutral").mean() * 100
            
            report_text += f"""
KEY METRICS
{'=' * 60}
Average Sentiment Score: {avg_sentiment:.3f}
Positive Comments: {positive_pct:.1f}%
Negative Comments: {negative_pct:.1f}%
Neutral Comments: {neutral_pct:.1f}%

KEY INSIGHTS
{'=' * 60}
"""
            
            insights = generate_insights(df_analyzed, video_info["title"])
            for insight in insights:
                safe_insight = sanitize_text_for_pdf(insight, method='remove')
                report_text += f"* {safe_insight}\n"
            
            report_text += f"""
SAMPLE COMMENTS
{'=' * 60}
"""
            
            for sentiment in ["Positive", "Neutral", "Negative"]:
                sentiment_comments = df_analyzed[df_analyzed["sentiment"] == sentiment].head(2)
                if len(sentiment_comments) > 0:
                    report_text += f"\n{sentiment.upper()} COMMENTS:\n"
                    for _, row in sentiment_comments.iterrows():
                        safe_comment = sanitize_text_for_pdf(row["comment"], method='remove')
                        truncated_comment = safe_comment[:80] + "..." if len(safe_comment) > 80 else safe_comment
                        report_text += f"  - {truncated_comment}\n"
            
            report_text += f"""
{'=' * 60}
Report generated by YouTube Sentiment Dashboard
"""
        
        return report_text.encode('utf-8', errors='replace')
        
    except Exception as e:
        return f"Error generating report: {str(e)}".encode('utf-8')

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_video_comments(youtube, video_id, min_comments=500):
    """
    Fetch comments from YouTube API with proper pagination to get at least min_comments
    """
    all_comments = []
    next_page_token = None
    
    while len(all_comments) < min_comments:
        try:
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
                
        except Exception as e:
            st.error(f"Error fetching comments: {str(e)}")
            break
    
    return all_comments

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
        
        # Fetch comments using the improved function
        all_comments = get_video_comments(youtube, video_id, min_comments=500)
        
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
        insights.append(f"Very Positive Reception: Comments show strong positive sentiment (avg: {avg_sentiment:.2f})")
    elif avg_sentiment > 0:
        insights.append(f"Generally Positive: Overall feedback is positive (avg: {avg_sentiment:.2f})")
    elif avg_sentiment < -0.2:
        insights.append(f"Strong Criticism: Significant negative feedback detected (avg: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        insights.append(f"Mixed with Concerns: Some negative feedback present (avg: {avg_sentiment:.2f})")
    else:
        insights.append(f"Neutral Dominance: Comments are mostly neutral or balanced")
    
    # Distribution insight
    dominant_sentiment = sentiment_counts.idxmax()
    dominant_percent = (sentiment_counts.max() / total_comments) * 100
    insights.append(f"{dominant_sentiment} Comments Dominate: {dominant_percent:.1f}% of all comments")
    
    # Top negative keywords insight
    if "Negative" in sentiment_counts:
        negative_comments = df[df["sentiment"] == "Negative"]["comment"]
        if len(negative_comments) > 0:
            negative_text = " ".join(negative_comments.str.lower())
            words = [word for word in negative_text.split() if len(word) > 3]
            common_words = Counter(words).most_common(3)
            if common_words:
                word_list = ", ".join([word for word, _ in common_words])
                insights.append(f"Common Concerns: Frequent words in negative comments: {word_list}")
    
    # Engagement insight
    if 'like_count' in df.columns:
        avg_likes = df["like_count"].mean()
        if avg_likes > 10:
            insights.append(f"High Engagement: Comments average {avg_likes:.0f} likes each")
        elif avg_likes > 5:
            insights.append(f"Good Engagement: Comments receive decent likes ({avg_likes:.0f} avg)")
    
    return insights

def create_comparison_chart(video_ids):
    """Create comparison chart for multiple videos"""
    if len(video_ids) < 2:
        return None
    
    comparison_data = []
    
    for vid in video_ids:
        if vid in st.session_state.video_data:
            data = st.session_state.video_data[vid]
            df = analyze_sentiment(data["df"].copy())
            
            total_comments = len(df)
            if total_comments == 0:
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

# ============================================
# MAIN APP LAYOUT
# ============================================

# Sidebar
st.sidebar.title("🎯 YouTube Sentiment")
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
                    with st.spinner('Fetching video data...'):
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
st.title("📊 YouTube Sentiment Analysis")

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
                            
                    
                        {row['comment'][:400]}{'...' if len(row['comment']) > 400 else ''}
                
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
                if st.button("📥 Download CSV", use_container_width=True, key="csv_btn"):
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Click to download",
                        data=csv,
                        file_name=f"youtube_sentiment_{selected_video}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="csv_download"
                    )
            
            with col2:
                if st.button("📊 Generate Report", use_container_width=True, key="report_btn"):
                    with st.spinner("Generating report..."):
                        try:
                            # Create summary
                            summary_data = {
                                "Metric": [
                                    "Video Title",
                                    "Total Comments",
                                    "Average Sentiment",
                                    "Positive Comments",
                                    "Negative Comments",
                                    "Neutral Comments",
                                    "Max Comment Likes",
                                    "Average Comment Length",
                                    "Analysis Date"
                                ],
                                "Value": [
                                    video_info["title"][:50] + "...",
                                    f"{len(df):,}",
                                    f"{df['sentiment_score'].mean():.3f}",
                                    f"{(df['sentiment'] == 'Positive').sum():,}",
                                    f"{(df['sentiment'] == 'Negative').sum():,}",
                                    f"{(df['sentiment'] == 'Neutral').sum():,}",
                                    f"{df['like_count'].max() if 'like_count' in df.columns else 0:,}",
                                    f"{df['comment'].str.len().mean():.1f}",
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                ]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            
                            st.markdown("### 📋 Analysis Summary")
                            st.dataframe(
                                summary_df,
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )
                            
                            # Show insights
                            st.markdown("### 🔍 Key Insights")
                            insights = generate_insights(df, video_info["title"])
                            for insight in insights:
                                st.info(insight)
                            
                            st.success("✅ Report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
            
            with col3:
                if st.button("📄 Generate PDF Report", use_container_width=True, key="pdf_btn"):
                    with st.spinner("Creating PDF report..."):
                        try:
                            pdf_bytes = generate_pdf_report(selected_video, video_info, df)
                            
                            if pdf_bytes:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"youtube_sentiment_report_{selected_video}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    key="pdf_download"
                                )
                                
                                st.success("✅ PDF report generated successfully!")
                            else:
                                st.error("Failed to generate PDF report")
                                
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                            # Fallback to text report
                            text_report = create_text_report_fallback(selected_video, video_info, df)
                            st.download_button(
                                label="📥 Download Text Report",
                                data=text_report,
                                file_name=f"sentiment_report_{selected_video}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key="txt_download"
                            )
            
            # Raw Data View
            st.markdown("### 📄 Raw Comment Data")
            st.dataframe(
                df[["comment", "sentiment", "sentiment_score", "published_at", "like_count", "author"]]
                .head(50)
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

