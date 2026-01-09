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
</style>
""", unsafe_allow_html=True)

# ============================================
# PDF SANITIZATION FUNCTION - ENHANCED VERSION
# ============================================
def sanitize_text_for_pdf(text, method='replace'):
    """
    Sanitize text for PDF generation to avoid encoding errors.
    """
    if text is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # If empty string, return as is
    if text.strip() == "":
        return text
    
    # Common replacements for problematic characters
    replacements = {
        '\u2022': '•',      # Bullet
        '\u25cf': '•',      # Black circle
        '\u25e6': '○',      # White circle
        '\u2013': '-',      # En dash
        '\u2014': '--',     # Em dash
        '\u2018': "'",      # Left single quotation
        '\u2019': "'",      # Right single quotation
        '\u201c': '"',      # Left double quotation
        '\u201d': '"',      # Right double quotation
        '\u00a9': '(c)',    # Copyright
        '\u00ae': '(R)',    # Registered
        '\u2122': '(TM)',   # Trademark
        '\u2026': '...',    # Ellipsis
        '\u2010': '-',      # Hyphen
        '\u2011': '-',      # Non-breaking hyphen
        '\u2012': '-',      # Figure dash
    }
    
    # Apply replacements
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    # Remove emojis and other non-ASCII characters
    if method == 'remove':
        # Remove all non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
    elif method == 'replace':
        # Replace non-ASCII with placeholder
        text = text.encode('ascii', 'replace').decode('ascii').replace('?', '')
    
    return text

# ============================================
# SAFE PDF GENERATION FUNCTION
# ============================================
def safe_generate_pdf(video_id, video_info, df):
    """
    Wrapper function that sanitizes all text before PDF generation
    """
    try:
        # Sanitize video info
        safe_video_info = {}
        for key, value in video_info.items():
            if key == 'df':
                continue  # Skip the dataframe
            if isinstance(value, dict):
                # Handle nested dictionaries (like stats)
                safe_value = {}
                for k, v in value.items():
                    safe_value[k] = sanitize_text_for_pdf(str(v))
                safe_video_info[key] = safe_value
            else:
                safe_video_info[key] = sanitize_text_for_pdf(str(value))
        
        # Sanitize DataFrame
        safe_df = df.copy()
        text_columns = safe_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            safe_df[col] = safe_df[col].apply(
                lambda x: sanitize_text_for_pdf(str(x)) if pd.notnull(x) else ""
            )
        
        # Generate PDF with sanitized data
        return generate_pdf_report(video_id, safe_video_info, safe_df)
        
    except Exception as e:
        st.error(f"Error in PDF generation: {str(e)}")
        raise

# ============================================
# UPDATED PDF GENERATION FUNCTION WITH UTF-8 FIX
# ============================================
def generate_pdf_report(video_id, video_info, df):
    """Generate a comprehensive PDF report with charts and insights"""
    
    # Analyze data
    df_analyzed = analyze_sentiment(df.copy())
    total_comments = len(df_analyzed)
    avg_sentiment = df_analyzed["sentiment_score"].mean()
    positive_pct = (df_analyzed["sentiment"] == "Positive").mean() * 100
    negative_pct = (df_analyzed["sentiment"] == "Negative").mean() * 100
    neutral_pct = (df_analyzed["sentiment"] == "Neutral").mean() * 100
    
    # Get insights
    insights = generate_insights(df_analyzed, video_info["title"])
    
    # Create PDF with UTF-8 support
    class UnicodePDF(FPDF):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add a Unicode compatible font
            self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
            self.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
            self.add_font('DejaVu', 'BI', 'DejaVuSans-BoldOblique.ttf', uni=True)
        
        def header(self):
            # Optional: Add header to each page
            pass
        
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Set font
            self.set_font('DejaVu', 'I', 8)
            # Page number
            self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
    
    # Create PDF instance
    pdf = UnicodePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- COVER PAGE ---
    pdf.add_page()
    
    # Set brand colors
    primary_color = (67, 97, 238)  # #4361ee
    secondary_color = (56, 176, 0)  # #38b000
    accent_color = (114, 9, 183)   # #7209b7
    
    # Google-style cover
    pdf.set_fill_color(*primary_color)
    pdf.rect(0, 0, 210, 50, 'F')
    
    # Title
    pdf.set_font('DejaVu', 'B', 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 60, 'YouTube Sentiment Analysis Report', 0, 1, 'C')
    
    # Video title
    pdf.set_font('DejaVu', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, '', 0, 1)  # Spacing
    
    # Video info box
    pdf.set_fill_color(245, 245, 245)
    pdf.rect(20, 80, 170, 60, 'F')
    pdf.set_xy(25, 85)
    pdf.set_font('DejaVu', 'B', 16)
    pdf.set_text_color(*primary_color)
    pdf.multi_cell(160, 8, sanitize_text_for_pdf(video_info.get("title", "Unknown Video"))[:80] + "...", 0, 'L')
    
    pdf.set_xy(25, 110)
    pdf.set_font('DejaVu', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(160, 8, f"Video ID: {video_id}", 0, 1, 'L')
    pdf.cell(160, 8, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.cell(160, 8, f"Total Comments Analyzed: {total_comments}", 0, 1, 'L')
    
    # --- EXECUTIVE SUMMARY PAGE ---
    pdf.add_page()
    
    # Page title
    pdf.set_fill_color(*primary_color)
    pdf.set_font('DejaVu', 'B', 20)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 20, 'Executive Summary', 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    
    # Key metrics
    pdf.set_font('DejaVu', 'B', 16)
    pdf.set_text_color(*secondary_color)
    pdf.cell(0, 15, 'Key Metrics', 0, 1, 'L')
    
    # Metrics grid
    metrics = [
        ("Total Comments", f"{total_comments:,}"),
        ("Avg Sentiment", f"{avg_sentiment:.3f}"),
        ("Positive Comments", f"{positive_pct:.1f}%"),
        ("Negative Comments", f"{negative_pct:.1f}%"),
        ("Neutral Comments", f"{neutral_pct:.1f}%")
    ]
    
    y_pos = 50
    for i, (label, value) in enumerate(metrics):
        x_pos = 10 + (i % 2) * 95
        if i % 2 == 0 and i > 0:
            y_pos += 25
        
        # Metric box
        pdf.set_fill_color(245, 245, 245)
        pdf.rect(x_pos, y_pos, 90, 20, 'F')
        
        # Label
        pdf.set_font('DejaVu', '', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.set_xy(x_pos + 5, y_pos + 3)
        pdf.cell(80, 5, label, 0, 1, 'L')
        
        # Value
        pdf.set_font('DejaVu', 'B', 16)
        pdf.set_text_color(*primary_color)
        pdf.set_xy(x_pos + 5, y_pos + 10)
        pdf.cell(80, 8, value, 0, 1, 'L')
    
    # --- AUTO INSIGHTS SECTION ---
    pdf.add_page()
    
    # Title
    pdf.set_fill_color(*accent_color)
    pdf.set_font('DejaVu', 'B', 20)
    pdf.set_text_color(*accent_color)
    pdf.cell(0, 20, 'Auto Insights & Recommendations', 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    
    # Insights
    y_pos = 40
    for insight in insights:
        if y_pos > 250:  # Check if we need new page
            pdf.add_page()
            y_pos = 20
        
        # Clean insight text
        clean_insight = sanitize_text_for_pdf(insight)
        
        # Insight box
        pdf.set_fill_color(240, 248, 255)
        pdf.rect(10, y_pos, 190, 15, 'F')
        
        # Insight text
        pdf.set_font('DejaVu', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(15, y_pos + 5)
        pdf.multi_cell(180, 5, clean_insight)
        
        y_pos += 20
    
    # --- SENTIMENT DISTRIBUTION ---
    pdf.add_page()
    
    # Title
    pdf.set_font('DejaVu', 'B', 20)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 20, 'Sentiment Distribution', 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    
    # Create simple chart (text-based)
    pdf.set_font('DejaVu', '', 12)
    pdf.set_text_color(100, 100, 100)
    
    # Positive bar
    pdf.set_fill_color(*secondary_color)
    pdf.rect(20, 50, positive_pct * 1.6, 15, 'F')
    pdf.set_xy(20 + positive_pct * 1.6 + 5, 53)
    pdf.cell(30, 8, f"Positive: {positive_pct:.1f}%", 0, 1, 'L')
    
    # Neutral bar
    pdf.set_fill_color(200, 200, 200)
    pdf.rect(20, 75, neutral_pct * 1.6, 15, 'F')
    pdf.set_xy(20 + neutral_pct * 1.6 + 5, 78)
    pdf.cell(30, 8, f"Neutral: {neutral_pct:.1f}%", 0, 1, 'L')
    
    # Negative bar
    pdf.set_fill_color(220, 53, 69)  # Red color
    pdf.rect(20, 100, negative_pct * 1.6, 15, 'F')
    pdf.set_xy(20 + negative_pct * 1.6 + 5, 103)
    pdf.cell(30, 8, f"Negative: {negative_pct:.1f}%", 0, 1, 'L')
    
    # --- DETAILED ANALYSIS ---
    pdf.add_page()
    
    # Title
    pdf.set_font('DejaVu', 'B', 20)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 20, 'Detailed Analysis', 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    
    # Sample comments
    pdf.set_font('DejaVu', 'B', 14)
    pdf.set_text_color(*accent_color)
    pdf.cell(0, 15, 'Sample Comments by Sentiment:', 0, 1, 'L')
    
    # Get sample comments
    sample_size = 5
    sentiments = ["Positive", "Neutral", "Negative"]
    
    y_pos = 60
    for sentiment in sentiments:
        sentiment_comments = df_analyzed[df_analyzed["sentiment"] == sentiment].head(sample_size)
        
        if len(sentiment_comments) > 0:
            # Sentiment header
            pdf.set_font('DejaVu', 'B', 12)
            color_map = {
                "Positive": secondary_color,
                "Neutral": (100, 100, 100),
                "Negative": (220, 53, 69)
            }
            pdf.set_text_color(*color_map[sentiment])
            pdf.cell(0, 10, f"{sentiment} Comments:", 0, 1, 'L')
            
            # Comments
            pdf.set_font('DejaVu', '', 10)
            pdf.set_text_color(0, 0, 0)
            
            for _, row in sentiment_comments.iterrows():
                if y_pos > 250:  # Check page space
                    pdf.add_page()
                    y_pos = 20
                
                # Ensure comment is properly sanitized
                safe_comment = sanitize_text_for_pdf(row["comment"])
                comment_text = safe_comment[:100] + "..." if len(safe_comment) > 100 else safe_comment
                pdf.multi_cell(0, 8, f"• {comment_text}")
                y_pos += 20
    
    # --- CONCLUSION ---
    pdf.add_page()
    
    # Title
    pdf.set_fill_color(*primary_color)
    pdf.set_font('DejaVu', 'B', 20)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 20, 'Conclusion & Next Steps', 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    
    # Recommendations based on sentiment
    pdf.set_font('DejaVu', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    recommendations = []
    if avg_sentiment > 0.2:
        recommendations = [
            "1. Leverage positive sentiment in marketing materials",
            "2. Engage with commenters to build community",
            "3. Consider creating similar content",
            "4. Share highlights on social media"
        ]
    elif avg_sentiment > 0:
        recommendations = [
            "1. Acknowledge positive feedback",
            "2. Address any recurring concerns",
            "3. Monitor sentiment trends",
            "4. Engage with constructive feedback"
        ]
    elif avg_sentiment < 0:
        recommendations = [
            "1. Review critical feedback carefully",
            "2. Consider making content adjustments",
            "3. Address common concerns in future content",
            "4. Monitor sentiment for improvement"
        ]
    else:
        recommendations = [
            "1. Seek more specific feedback",
            "2. Encourage viewer engagement",
            "3. Test different content approaches",
            "4. Monitor for sentiment shifts"
        ]
    
    y_pos = 40
    for rec in recommendations:
        pdf.set_xy(15, y_pos)
        pdf.cell(0, 10, rec, 0, 1, 'L')
        y_pos += 12
    
    # Footer
    pdf.set_y(-30)
    pdf.set_font('DejaVu', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Generated by YouTube Sentiment Analysis Dashboard', 0, 0, 'C')
    
    # Return PDF as bytes - FIXED ENCODING ISSUE
    try:
        # Method 1: Try to get as bytes directly
        pdf_bytes = pdf.output(dest='S')
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin-1', errors='replace')
        return pdf_bytes
    except:
        # Method 2: Fallback to BytesIO
        try:
            pdf_bytes = pdf.output()
            return pdf_bytes
        except:
            # Method 3: Ultimate fallback
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    pdf.output(tmp.name)
                    with open(tmp.name, 'rb') as f:
                        pdf_bytes = f.read()
                return pdf_bytes
            except Exception as e:
                st.error(f"Final PDF generation error: {str(e)}")
                return b"PDF generation failed"

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
                maxResults=100,  # API maximum per request
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
            
            # Check for more pages
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                # No more comments available
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

# ============================================
# REST OF THE ORIGINAL CODE (UNCHANGED)
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
                    with st.spinner(''):
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
                    <div style="margin-top: 10px; line-height: 1.6; color: #333;">
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
                            # Create summary without mixing types in same DataFrame
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
                            # Display the summary in a nicer format
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
                            # Use the safe PDF generation function
                            pdf_bytes = safe_generate_pdf(selected_video, video_info, df)
                            
                            # Create sanitized filename
                            safe_filename = sanitize_text_for_pdf(
                                f"youtube_sentiment_report_{selected_video}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            )
                            
                            # Create download button
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_bytes,
                                file_name=safe_filename,
                                mime="application/pdf",
                                use_container_width=True,
                                key="pdf_download"
                            )
                            
                            st.success("✅ PDF report generated successfully!")
                            
                            # Show preview of what's in the PDF
                            with st.expander("📋 PDF Contents Preview"):
                                st.markdown("""
                                **Report includes:**
                                - Google-style cover page with video info
                                - Executive summary with key metrics
                                - Auto-generated insights section
                                - Sentiment distribution charts
                                - Sample comments by sentiment
                                - Conclusions and recommendations
                                - Brand colors throughout
                                """)
                                
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
            
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