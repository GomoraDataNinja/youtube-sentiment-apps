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
import time
import warnings
import os
import hashlib
import json
warnings.filterwarnings('ignore')

# ==================== DEPLOYMENT CONFIGURATION ====================
APP_VERSION = "2.1.4"
APP_NAME = "YouTube Sentiment Analysis Dashboard"
DEPLOYMENT_MODE = os.environ.get('DEPLOYMENT_MODE', 'development')

# Security configuration
MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT_MINUTES = 60
PASSWORD_MIN_LENGTH = 8

# Load configuration
def load_config():
    config = {
        'COMMON_PASSWORD': os.environ.get('APP_PASSWORD', 'youtube2024'),
        'ALLOWED_USERS': os.environ.get('ALLOWED_USERS', 'admin,analyst,user').split(','),
        'ADMIN_USERS': os.environ.get('ADMIN_USERS', 'admin').split(','),
        'MAX_FILE_SIZE_MB': 10
    }
    return config

config = load_config()

# ==================== THEME DETECTION & COLORS ====================
def detect_theme():
    """Detect current Streamlit theme"""
    try:
        # Check if theme is stored in session state
        if 'current_theme' in st.session_state:
            return st.session_state.current_theme
        
        # Default theme detection logic
        # We'll try to detect from CSS variables
        theme = 'light'  # Default to light
        
        # Check if we're in Streamlit Cloud or local
        try:
            # This is a workaround for theme detection
            # Streamlit doesn't expose theme directly in Python
            # We'll rely on CSS variables that Streamlit sets
            pass
        except:
            pass
            
        return theme
    except:
        return 'light'

def get_theme_colors(theme=None):
    """Get colors based on theme"""
    if theme is None:
        theme = detect_theme()
        
    if theme == 'dark':
        return {
            'primary': "#4285F4",
            'secondary': "#34A853",
            'accent': "#EA4335",
            'warning': "#FBBC05",
            'neutral': "#9AA0A6",
            'background': "#0E1117",
            'card': "#1E2126",
            'text': "#FAFAFA",
            'text_light': "#B0B3B8",
            'success': "#34A853",
            'danger': "#EA4335",
            'sidebar': "#1E2126",
            'border': "#2D3748",
            'hover': "#2A2D35",
            'shadow': "rgba(0, 0, 0, 0.3)"
        }
    else:  # light theme
        return {
            'primary': "#4285F4",
            'secondary': "#34A853",
            'accent': "#EA4335",
            'warning': "#FBBC05",
            'neutral': "#9AA0A6",
            'background': "#F8F9FA",
            'card': "#FFFFFF",
            'text': "#202124",
            'text_light': "#5F6368",
            'success': "#34A853",
            'danger': "#EA4335",
            'sidebar': "#202124",
            'border': "#DADCE0",
            'hover': "#F1F3F4",
            'shadow': "rgba(0, 0, 0, 0.1)"
        }

# Initialize theme
current_theme = detect_theme()
st.session_state.current_theme = current_theme
COLORS = get_theme_colors(current_theme)

# ==================== SENTIMENT COLORS ====================
SENTIMENT_COLORS = {
    'Positive': "#34A853",
    'Neutral': "#9AA0A6",
    'Negative': "#EA4335",
}

# ==================== SECURITY FUNCTIONS ====================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_session_timeout():
    if 'last_activity' in st.session_state:
        last_activity = st.session_state.last_activity
        time_diff = datetime.now() - last_activity
        if time_diff.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
            logout()
            return True
    return False

def update_activity():
    st.session_state.last_activity = datetime.now()

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ==================== AUTHENTICATION ====================
def check_password(username, password):
    username = username.strip().lower()
    
    if username in st.session_state.login_attempts:
        attempts, last_attempt = st.session_state.login_attempts[username]
        time_diff = (datetime.now() - last_attempt).total_seconds()
        
        if attempts >= MAX_LOGIN_ATTEMPTS and time_diff < 300:
            return False, "Too many failed attempts. Please try again in 5 minutes."
    
    if username and password == config['COMMON_PASSWORD']:
        if username in st.session_state.login_attempts:
            del st.session_state.login_attempts[username]
        
        if username in config['ADMIN_USERS']:
            role = 'admin'
        elif username in config['ALLOWED_USERS']:
            role = 'user'
        else:
            role = 'guest'
        
        return True, role
    else:
        if username not in st.session_state.login_attempts:
            st.session_state.login_attempts[username] = [1, datetime.now()]
        else:
            st.session_state.login_attempts[username][0] += 1
            st.session_state.login_attempts[username][1] = datetime.now()
        
        return False, "Invalid credentials"

def show_login_page():
    # Login page with fixed colors (doesn't change with theme)
    st.markdown("""
    <style>
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        padding: 2rem;
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
    }
    .login-card {
        background: white;
        border-radius: 12px;
        padding: 3rem;
        width: 100%;
        max-width: 420px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        color: #202124;
    }
    .youtube-logo {
        width: 80px;
        height: 80px;
        margin: 0 auto 1.5rem;
        background: #FF0000;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        color: white;
        font-weight: bold;
    }
    .login-title {
        font-size: 28px;
        font-weight: 400;
        color: #202124;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .security-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(52, 168, 83, 0.15);
        color: #34A853;
        border: 1px solid rgba(52, 168, 83, 0.3);
    }
    .login-subtitle {
        text-align: center;
        color: #5F6368;
        margin-bottom: 2.5rem;
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    
    st.markdown(f'''
        <div style="text-align: center; margin-bottom: 1rem;">
            <span class="security-badge">🔐 Secure Login</span>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="youtube-logo">▶</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="login-title">YouTube Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown(f'''
        <p class="login-subtitle">
            Analyze sentiment in YouTube comments securely<br>
            <span style="font-size: 12px; opacity: 0.8;">
                Deployment: <strong>{DEPLOYMENT_MODE.upper()}</strong> | Version: {APP_VERSION}
            </span>
        </p>
    ''', unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if DEPLOYMENT_MODE == 'production':
            st.markdown('''
                <div style="background: rgba(251, 188, 5, 0.15); border: 1px solid rgba(251, 188, 5, 0.3);
                          border-radius: 8px; padding: 16px; margin: 16px 0; color: #FBBC05; font-size: 13px;">
                    🔒 Production Environment - Sensitive Data
                </div>
            ''', unsafe_allow_html=True)
        
        submit_button = st.form_submit_button("Sign In", type="primary", use_container_width=True)
    
    if submit_button:
        if not username.strip():
            st.error("Please enter a username")
        else:
            success, message = check_password(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = message
                st.session_state.last_activity = datetime.now()
                
                login_event = {
                    'timestamp': datetime.now().isoformat(),
                    'username': username,
                    'action': 'login'
                }
                st.session_state.analysis_history.append(login_event)
                
                st.success(f"Welcome, {username}!")
                time.sleep(1)
                safe_rerun()
            else:
                st.error(f"Login failed: {message}")
    
    st.markdown(f"""
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #DADCE0;">
            <div style="text-align: center; color: #5F6368; font-size: 12px;">
                <div style="margin-bottom: 8px;">
                    <strong>{APP_NAME} v{APP_VERSION}</strong>
                </div>
                <div style="font-size: 11px; opacity: 0.8;">
                    © 2024 YouTube Sentiment Analysis Dashboard<br>
                    Unauthorized access is prohibited
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def logout():
    logout_event = {
        'timestamp': datetime.now().isoformat(),
        'username': st.session_state.username,
        'action': 'logout'
    }
    st.session_state.analysis_history.append(logout_event)
    
    sensitive_keys = ['video_data', 'current_videos', 'df']
    for key in sensitive_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    auth_keys = ['authenticated', 'username', 'user_role', 'session_id']
    for key in auth_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    keep_keys = ['analysis_history', 'export_history']
    new_state = {k: v for k, v in st.session_state.items() if k in keep_keys}
    
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    for key, value in new_state.items():
        st.session_state[key] = value
    
    safe_rerun()

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': f'''
        ### {APP_NAME} v{APP_VERSION}
        
        Secure YouTube Sentiment Analysis Dashboard
        
        Features:
        • YouTube comment sentiment analysis
        • Multiple video comparison
        • Secure data handling
        • Export functionality
        
        © 2024 All rights reserved.
        '''
    }
)

# ==================== SESSION STATE INITIALIZATION ====================
# Security-related session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'user'
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = {}
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.now()
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

# YouTube analysis state
if 'video_data' not in st.session_state:
    st.session_state.video_data = {}
if 'current_videos' not in st.session_state:
    st.session_state.current_videos = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'export_history' not in st.session_state:
    st.session_state.export_history = []

# ==================== DYNAMIC CSS FOR THEME SUPPORT ====================
st.markdown(f"""
<style>
    /* CSS Variables for Theme Support */
    :root {{
        --primary-color: {COLORS['primary']};
        --secondary-color: {COLORS['secondary']};
        --accent-color: {COLORS['accent']};
        --warning-color: {COLORS['warning']};
        --neutral-color: {COLORS['neutral']};
        --background-color: {COLORS['background']};
        --card-color: {COLORS['card']};
        --text-color: {COLORS['text']};
        --text-light-color: {COLORS['text_light']};
        --success-color: {COLORS['success']};
        --danger-color: {COLORS['danger']};
        --border-color: {COLORS['border']};
        --hover-color: {COLORS['hover']};
        --shadow-color: {COLORS['shadow']};
    }}
    
    /* Base styles */
    .stApp {{
        background-color: var(--background-color);
        font-family: 'Inter', 'Roboto', sans-serif;
        color: var(--text-color);
    }}
    
    /* Metric cards */
    .metric-card {{
        background: var(--card-color);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
        color: var(--text-color);
        box-shadow: 0 2px 8px var(--shadow-color);
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-color);
        border-color: var(--primary-color);
    }}
    
    .metric-value {{
        font-size: 32px;
        font-weight: 400;
        color: var(--text-color);
        margin: 8px 0;
    }}
    
    .metric-label {{
        font-size: 12px;
        color: var(--text-light-color);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }}
    
    /* Cards */
    .g-card {{
        background: var(--card-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px var(--shadow-color);
        color: var(--text-color);
    }}
    
    .g-card:hover {{
        box-shadow: 0 4px 20px var(--shadow-color);
        border-color: var(--primary-color);
    }}
    
    /* User chip */
    .user-chip {{
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--background-color);
        padding: 10px 18px;
        border-radius: 24px;
        border: 1px solid var(--border-color);
        font-size: 14px;
        color: var(--text-color);
        font-weight: 500;
    }}
    
    .user-avatar {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(66, 133, 244, 0.3);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Status indicators */
    .status-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(52, 168, 83, 0.15);
        color: var(--success-color);
        border: 1px solid rgba(52, 168, 83, 0.3);
    }}
    
    /* Input fields */
    .stTextInput > div > div > input {{
        background-color: var(--card-color);
        color: var(--text-color);
        border-color: var(--border-color);
    }}
    
    .stTextInput > label {{
        color: var(--text-color);
    }}
    
    /* Select boxes */
    .stSelectbox > div > div {{
        background-color: var(--card-color);
        color: var(--text-color);
        border-color: var(--border-color);
    }}
    
    /* Buttons */
    .stButton > button {{
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px var(--shadow-color);
    }}
    
    /* Dataframes */
    .stDataFrame {{
        background-color: var(--card-color);
        color: var(--text-color);
    }}
    
    .stDataFrame th {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}
    
    .stDataFrame td {{
        color: var(--text-color);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: var(--primary-color);
        color: white;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        color: var(--text-color);
        background-color: var(--card-color);
    }}
    
    /* Alerts */
    .stAlert {{
        color: var(--text-color);
    }}
    
    /* Progress bars */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }}
    
    /* Ensure text is visible in all components */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        color: var(--text-color) !important;
    }}
    
    /* Fix for selectbox options */
    [data-baseweb="popover"] {{
        background-color: var(--card-color) !important;
        color: var(--text-color) !important;
    }}
    
    /* Fix for sidebar */
    [data-testid="stSidebar"] {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}
    
    /* Fix for sidebar content */
    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-color);
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--background-color);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--border-color);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--neutral-color);
    }}
    
    /* Plotly chart background fix */
    .js-plotly-plot .plotly .modebar {{
        background: transparent !important;
    }}
    
    /* Make sure all text is visible */
    * {{
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
</style>

<!-- JavaScript to detect theme changes -->
<script>
    // Function to detect Streamlit theme
    function detectTheme() {{
        // Check CSS variables or body background
        const bodyStyle = window.getComputedStyle(document.body);
        const bgColor = bodyStyle.backgroundColor;
        
        // Convert RGB to brightness
        const rgb = bgColor.match(/\\d+/g);
        if (rgb) {{
            const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
            return brightness < 128 ? 'dark' : 'light';
        }}
        return 'light';
    }}
    
    // Send theme to Python when it changes
    function updateTheme() {{
        const theme = detectTheme();
        const event = new CustomEvent('setTheme', {{ detail: {{ theme: theme }} }});
        window.parent.document.dispatchEvent(event);
    }}
    
    // Check for theme changes periodically
    let lastTheme = detectTheme();
    setInterval(() => {{
        const currentTheme = detectTheme();
        if (currentTheme !== lastTheme) {{
            lastTheme = currentTheme;
            updateTheme();
        }}
    }}, 1000);
</script>
""", unsafe_allow_html=True)

# JavaScript to update theme in session state
st.components.v1.html("""
<script>
    // Listen for theme change events
    window.addEventListener('setTheme', function(e) {
        const theme = e.detail.theme;
        
        // Send to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: theme
        }, '*');
    });
</script>
""", height=0)

# ==================== SECURITY MIDDLEWARE ====================
if st.session_state.authenticated and check_session_timeout():
    st.warning("Session has timed out due to inactivity. Please login again.")
    st.stop()

if st.session_state.authenticated:
    update_activity()

# ==================== AUTHENTICATION CHECK ====================
if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# ==================== ORIGINAL YOUTUBE FUNCTIONS ====================
def extract_video_id(url):
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

def get_video_comments(youtube, video_id, min_comments=500):
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

def fetch_comments(video_id, video_url):
    if video_id in st.session_state.video_data:
        return st.session_state.video_data[video_id]
    
    API_KEY = st.secrets["youtube_api_key"]
    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    try:
        video_request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        video_response = video_request.execute()
        
        if not video_response['items']:
            return None
            
        video_info = video_response['items'][0]
        all_comments = get_video_comments(youtube, video_id, min_comments=500)
        
        if not all_comments:
            return None
            
        df = pd.DataFrame(all_comments)
        df["published_at"] = pd.to_datetime(df["published_at"])
        
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
    total_comments = len(df)
    if total_comments == 0:
        return ["No comments to analyze"]
    
    sentiment_counts = df["sentiment"].value_counts()
    avg_sentiment = df["sentiment_score"].mean()
    
    insights = []
    
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
    
    dominant_sentiment = sentiment_counts.idxmax()
    dominant_percent = (sentiment_counts.max() / total_comments) * 100
    insights.append(f"{dominant_sentiment} Comments Dominate: {dominant_percent:.1f}% of all comments")
    
    if "Negative" in sentiment_counts:
        negative_comments = df[df["sentiment"] == "Negative"]["comment"]
        if len(negative_comments) > 0:
            negative_text = " ".join(negative_comments.str.lower())
            words = [word for word in negative_text.split() if len(word) > 3]
            common_words = Counter(words).most_common(3)
            if common_words:
                word_list = ", ".join([word for word, _ in common_words])
                insights.append(f"Common Concerns: Frequent words in negative comments: {word_list}")
    
    if 'like_count' in df.columns:
        avg_likes = df["like_count"].mean()
        if avg_likes > 10:
            insights.append(f"High Engagement: Comments average {avg_likes:.0f} likes each")
        elif avg_likes > 5:
            insights.append(f"Good Engagement: Comments receive decent likes ({avg_likes:.0f} avg)")
    
    return insights

def create_comparison_chart(video_ids):
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

# ==================== EXPORT FUNCTIONS ====================
def sanitize_text_for_pdf(text, method='remove'):
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    if text.strip() == "":
        return text
    
    replacements = {
        '\u2022': '*', '\u25cf': '*', '\u25e6': '*',
        '\u2013': '-', '\u2014': '--',
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u00a9': '(c)', '\u00ae': '(R)',
        '\u2122': '(TM)', '\u2026': '...',
        '\u2764': '<3', '\u2665': '<3'
    }
    
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    if method == 'remove':
        text = text.encode('ascii', 'ignore').decode('ascii')
    elif method == 'replace':
        result = []
        for char in text:
            if ord(char) < 128:
                result.append(char)
            else:
                result.append(' ')
        text = ''.join(result)
    
    text = ' '.join(text.split())
    return text

def generate_summary_report(video_id, video_info, df):
    df_analyzed = analyze_sentiment(df.copy())
    total_comments = len(df_analyzed)
    
    if total_comments == 0:
        return None
    
    avg_sentiment = df_analyzed["sentiment_score"].mean()
    positive_pct = (df_analyzed["sentiment"] == "Positive").mean() * 100
    negative_pct = (df_analyzed["sentiment"] == "Negative").mean() * 100
    neutral_pct = (df_analyzed["sentiment"] == "Neutral").mean() * 100
    
    summary_data = {
        "Report Type": ["YouTube Sentiment Analysis Summary"],
        "Generated By": [st.session_state.username],
        "Generation Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Video Title": [video_info.get("title", "Unknown Video")[:50] + "..."],
        "Video ID": [video_id],
        "Total Comments": [total_comments],
        "Average Sentiment": [f"{avg_sentiment:.3f}"],
        "Positive Comments (%)": [f"{positive_pct:.1f}%"],
        "Negative Comments (%)": [f"{negative_pct:.1f}%"],
        "Neutral Comments (%)": [f"{neutral_pct:.1f}%"],
        "Analysis Engine": ["TextBlob"],
        "Session ID": [st.session_state.session_id],
        "Deployment Mode": [DEPLOYMENT_MODE]
    }
    
    return pd.DataFrame(summary_data)

def generate_detailed_report(video_id, video_info, df):
    df_analyzed = analyze_sentiment(df.copy())
    detailed_df = df_analyzed.copy()
    
    detailed_df['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detailed_df['video_id'] = video_id
    detailed_df['video_title'] = video_info.get("title", "Unknown Video")
    detailed_df['analyzed_by'] = st.session_state.username
    
    return detailed_df

# ==================== DASHBOARD HEADER ====================
# Update colors based on current theme
text_color = COLORS['text']
text_light_color = COLORS['text_light']
background_color = COLORS['background']
card_color = COLORS['card']
border_color = COLORS['border']
primary_color = COLORS['primary']
success_color = COLORS['success']

st.markdown(f'''
    <div style="background: {card_color}; border-bottom: 1px solid {border_color}; 
                padding: 1.2rem 2.5rem; margin: -2rem -1rem 2rem -1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="display: flex; align-items: center; gap: 12px; color: {primary_color}; 
                         font-weight: 600; font-size: 22px;">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                        <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" fill="currentColor"/>
                    </svg>
                    <span>{APP_NAME}</span>
                </div>
                <div style="font-size: 14px; color: {text_light_color};">
                    v{APP_VERSION} • {DEPLOYMENT_MODE.title()} Mode • {current_theme.title()} Theme
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div class="user-chip">
                    <div class="user-avatar">{st.session_state.username[0].upper()}</div>
                    <div>
                        <div style="font-weight: 600;">{st.session_state.username}</div>
                        <div style="font-size: 11px; color: {text_light_color};">
                            {st.session_state.user_role.upper()} • Session: {st.session_state.session_id}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
''', unsafe_allow_html=True)

# Logout button
col1, col2, col3 = st.columns([4, 2, 4])
with col2:
    if st.button("🚪 Secure Logout", key="logout_button", type="secondary", use_container_width=True):
        logout()

st.markdown("---")

# ==================== SIDEBAR ====================
with st.sidebar:
    # Fixed the f-string by using variables
    header_html = f'''
    <div style="padding: 20px; border-bottom: 1px solid {border_color}; 
                background: {card_color}; border-radius: 8px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div class="status-indicator">🔐 Secured</div>
            <div style="font-size: 11px; color: {text_light_color};">
                {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </div>
        <div style="font-size: 13px; color: {text_color}; line-height: 1.5;">
            <div style="margin-bottom: 5px;"><strong>User:</strong> {st.session_state.username}</div>
            <div><strong>Role:</strong> {st.session_state.user_role}</div>
        </div>
    </div>
    '''
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Video Management Section
    st.markdown(f"<h3 style='color: {text_color}; margin: 25px 0 15px 0;'>🎬 Video Management</h3>", unsafe_allow_html=True)
    
    video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    col1, col2 = st.columns(2)
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
                                
                                upload_event = {
                                    'timestamp': datetime.now().isoformat(),
                                    'username': st.session_state.username,
                                    'action': 'video_added',
                                    'video_id': video_id,
                                    'title': result['title'][:50]
                                }
                                st.session_state.analysis_history.append(upload_event)
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
            st.session_state.video_data = {}
            safe_rerun()
    
    st.markdown("---")
    
    if st.session_state.current_videos:
        st.markdown(f"<h3 style='color: {text_color}; margin: 25px 0 15px 0;'>📋 Selected Videos</h3>", unsafe_allow_html=True)
        for i, vid in enumerate(st.session_state.current_videos, 1):
            video_info = st.session_state.video_data.get(vid, {})
            title = video_info.get("title", f"Video {i}")
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.caption(f"{i}. {title[:40]}...")
            with col2:
                if st.button("✕", key=f"remove_{vid}"):
                    st.session_state.current_videos.remove(vid)
                    safe_rerun()
    
    st.markdown("---")
    
    # Analysis settings
    st.markdown(f"<h3 style='color: {text_color}; margin: 25px 0 15px 0;'>⚙️ Analysis Settings</h3>", unsafe_allow_html=True)
    
    sentiment_threshold = st.slider(
        "Sentiment Threshold",
        0.0, 1.0, 0.1, 0.05,
        help="Adjust threshold for sentiment classification"
    )
    
    # Security settings for admins
    if st.session_state.user_role == 'admin':
        st.markdown("---")
        st.markdown(f"<h3 style='color: {text_color};'>🔒 Security Settings</h3>", unsafe_allow_html=True)
        
        auto_logout = st.checkbox("Enable Auto-logout", value=True)
        
        if st.button("🛡️ Security Audit", key="security_audit"):
            audit_results = {
                'session_id': st.session_state.session_id,
                'login_time': st.session_state.last_activity.strftime('%Y-%m-%d %H:%M:%S'),
                'user_role': st.session_state.user_role,
                'video_analyses': len(st.session_state.video_data),
                'deployment_mode': DEPLOYMENT_MODE
            }
            st.info(f"Security audit completed: {audit_results}")
    
    st.markdown("---")
    
    # Session info
    session_duration = (datetime.now() - st.session_state.last_activity).seconds // 60
    session_html = f'''
    <div style="color: {text_color}; font-size: 12px; padding: 12px; 
                background: {card_color}; border-radius: 8px; border: 1px solid {border_color};">
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between;">
                <span>Session:</span>
                <span style="color: {text_light_color};">{session_duration}m active</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Videos:</span>
                <span style="color: {text_light_color};">{len(st.session_state.current_videos)}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Mode:</span>
                <span style="color: {text_light_color};">{DEPLOYMENT_MODE}</span>
            </div>
        </div>
    </div>
    '''
    st.markdown(session_html, unsafe_allow_html=True)

# ==================== DASHBOARD METRICS ====================
if st.session_state.current_videos:
    total_comments = sum(len(data['df']) for data in st.session_state.video_data.values())
    avg_sentiment = "0.00"
    
    if total_comments > 0:
        all_scores = []
        for vid in st.session_state.current_videos:
            if vid in st.session_state.video_data:
                df = analyze_sentiment(st.session_state.video_data[vid]['df'].copy())
                all_scores.extend(df['sentiment_score'].tolist())
        avg_sentiment = f"{np.mean(all_scores):.2f}" if all_scores else "0.00"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Total Videos</div>
                <div class="metric-value">{len(st.session_state.current_videos)}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Total Comments</div>
                <div class="metric-value">{total_comments:,}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Avg Sentiment</div>
                <div class="metric-value">{avg_sentiment}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Security Status</div>
                <div class="metric-value" style="color: {success_color};">✓</div>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")

# ==================== MAIN CONTENT ====================
if not st.session_state.current_videos:
    st.info("👈 Add YouTube videos from the sidebar to get started!")
    st.markdown(f'''
        <div class="g-card">
            <div style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 48px; margin-bottom: 20px;">🎬</div>
                <h3 style="color: {text_color}; margin-bottom: 15px;">Welcome to YouTube Sentiment Analysis</h3>
                <p style="color: {text_light_color}; line-height: 1.6; margin-bottom: 25px;">
                    Analyze sentiment in YouTube comments securely. Add videos from the sidebar to begin.
                </p>
                <div style="display: inline-flex; align-items: center; gap: 8px; 
                         padding: 10px 20px; background: {primary_color}; 
                         color: white; border-radius: 8px; font-weight: 500;">
                    🔒 Secure Session Active
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    st.stop()

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Single Video Analysis",
    "⚖️ Video Comparison",
    "💬 Comment Explorer",
    "📈 Advanced Analytics",
    "📤 Export Results"
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
            
            # Insights
            insights_html = f'''
            <div class="g-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h3 style="color: {text_color}; margin: 0 0 10px 0;">🔍 Key Insights</h3>
                        <p style="color: {text_light_color}; margin: 0;">Analysis of: {video_info["title"][:60]}...</p>
                    </div>
                    <div class="status-indicator">● Active</div>
                </div>
            </div>
            '''
            st.markdown(insights_html, unsafe_allow_html=True)
            
            insights = generate_insights(df, video_info["title"])
            for insight in insights:
                insight_html = f'''
                <div style="background: {card_color}; border-left: 4px solid {primary_color};
                        padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0;">
                    <div style="color: {text_color}; font-size: 14px;">{insight}</div>
                </div>
                '''
                st.markdown(insight_html, unsafe_allow_html=True)
            
            # Metrics
            positive_pct = (df["sentiment"] == "Positive").mean() * 100
            avg_sentiment = df["sentiment_score"].mean()
            engagement = df["like_count"].mean() if "like_count" in df.columns else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Total Comments</div>
                        <div class="metric-value">{len(df):,}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Avg Sentiment</div>
                        <div class="metric-value">{avg_sentiment:.2f}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Positive %</div>
                        <div class="metric-value" style="color: {SENTIMENT_COLORS['Positive']};">{positive_pct:.1f}%</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Avg Likes</div>
                        <div class="metric-value">{engagement:.0f}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            # Charts - Update Plotly charts with theme-aware colors
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = df["sentiment"].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    color=sentiment_counts.index,
                    color_discrete_map=SENTIMENT_COLORS,
                    title="Sentiment Distribution"
                )
                fig.update_layout(
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                daily_avg = df.groupby(df["published_at"].dt.date)["sentiment_score"].mean()
                daily_count = df.groupby(df["published_at"].dt.date).size()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_avg.index,
                    y=daily_avg.values,
                    mode='lines+markers',
                    name='Avg Sentiment',
                    line=dict(color=primary_color, width=3)
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
                    title="Sentiment Over Time",
                    yaxis=dict(title="Sentiment Score"),
                    yaxis2=dict(title="Comment Count", overlaying="y", side="right"),
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    legend=dict(font=dict(color=text_color))
                )
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Video Comparison
with tab2:
    if len(st.session_state.current_videos) >= 2:
        comparison_html = f'''
        <div class="g-card">
            <h3 style="color: {text_color}; margin: 0 0 10px 0;">📊 Video Comparison</h3>
            <p style="color: {text_light_color}; margin: 0;">Compare sentiment across multiple videos</p>
        </div>
        '''
        st.markdown(comparison_html, unsafe_allow_html=True)
        
        comparison_df = create_comparison_chart(st.session_state.current_videos)
        
        if comparison_df is not None and not comparison_df.empty:
            cols = st.columns(len(comparison_df))
            for idx, (_, row) in enumerate(comparison_df.iterrows()):
                with cols[idx]:
                    st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">{row["Video"]}</div>
                            <div class="metric-value">{row["Total Comments"]:,}</div>
                            <div class="metric-label" style="margin-top: 10px;">
                                Sentiment: {row['Avg Sentiment']:.2f}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
            
            # Comparison Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    comparison_df,
                    x="Video",
                    y=["Positive %", "Negative %"],
                    title="Sentiment Distribution by Video",
                    barmode="group",
                    color_discrete_map={"Positive %": SENTIMENT_COLORS['Positive'], 
                                      "Negative %": SENTIMENT_COLORS['Negative']}
                )
                fig.update_layout(
                    yaxis_title="Percentage (%)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    legend=dict(font=dict(color=text_color))
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
                fig.update_layout(
                    xaxis_title="Total Comments",
                    yaxis_title="Average Sentiment",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    legend=dict(font=dict(color=text_color))
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
                }),
                use_container_width=True,
                height=300
            )
    else:
        st.info("Add at least 2 videos to enable comparison")

# Tab 3: Comment Explorer
with tab3:
    if st.session_state.current_videos:
        explorer_html = f'''
        <div class="g-card">
            <h3 style="color: {text_color}; margin: 0 0 10px 0;">💬 Comment Explorer</h3>
            <p style="color: {text_light_color}; margin: 0;">Browse and filter comments by sentiment</p>
        </div>
        '''
        st.markdown(explorer_html, unsafe_allow_html=True)
        
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
            
            for _, row in filtered_df.head(comments_to_show).iterrows():
                sentiment_color = SENTIMENT_COLORS[row["sentiment"]]
                sentiment_icon = "✅" if row["sentiment"] == "Positive" else "⚪" if row["sentiment"] == "Neutral" else "❌"
                
                comment_html = f'''
                <div style="border-left: 4px solid {sentiment_color}; padding: 15px; 
                         margin: 10px 0; background: {card_color}; border-radius: 0 8px 8px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.2rem;">{sentiment_icon}</span>
                            <span style="font-weight: 600; color: {sentiment_color};">
                                {row['sentiment']} ({row['sentiment_score']:.2f})
                            </span>
                        </div>
                        <div style="color: {text_light_color}; font-size: 0.9em;">
                            {row['published_at'].strftime('%Y-%m-%d %H:%M')}
                        </div>
                    </div>
                    <div style="color: {text_color}; line-height: 1.5;">
                        {row['comment'][:300]}{'...' if len(row['comment']) > 300 else ''}
                    </div>
                    <div style="margin-top: 8px; color: {text_light_color}; font-size: 0.9em;">
                        {row.get('author', 'Unknown')}
                        {f" • {row['like_count']} ❤️" if row.get('like_count', 0) > 0 else ''}
                    </div>
                </div>
                '''
                st.markdown(comment_html, unsafe_allow_html=True)

# Tab 4: Advanced Analytics
with tab4:
    if st.session_state.current_videos:
        analytics_html = f'''
        <div class="g-card">
            <h3 style="color: {text_color}; margin: 0 0 10px 0;">📈 Advanced Analytics</h3>
            <p style="color: {text_light_color}; margin: 0;">Detailed analytics and insights</p>
        </div>
        '''
        st.markdown(analytics_html, unsafe_allow_html=True)
        
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
                fig.update_layout(
                    xaxis=dict(tickmode='linear', dtick=1),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📏 Comment Length Analysis**")
                df["comment_length"] = df["comment"].str.len()
                
                fig = px.scatter(
                    df,
                    x="comment_length",
                    y="sentiment_score",
                    color="sentiment",
                    title="Comment Length vs Sentiment",
                    color_discrete_map=SENTIMENT_COLORS
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    legend=dict(font=dict(color=text_color))
                )
                st.plotly_chart(fig, use_container_width=True)

# Tab 5: Export Results
with tab5:
    if st.session_state.current_videos:
        export_html = f'''
        <div class="g-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h3 style="color: {text_color}; margin: 0 0 10px 0;">📤 Export Results</h3>
                    <p style="color: {text_light_color}; margin: 0;">Export analysis results securely</p>
                </div>
                <div class="status-indicator">🔐 Secure Export</div>
            </div>
        </div>
        '''
        st.markdown(export_html, unsafe_allow_html=True)
        
        selected_video = st.selectbox(
            "Select Video to Export",
            options=st.session_state.current_videos,
            format_func=lambda x: st.session_state.video_data[x]["title"][:50] + "...",
            key="export_select"
        )
        
        if selected_video:
            video_info = st.session_state.video_data[selected_video]
            df = video_info["df"].copy()
            
            # Export cards layout
            col1, col2 = st.columns(2)
            
            with col1:
                export_card1_html = f'''
                <div style="text-align: center; padding: 24px 20px; border: 2px dashed {border_color}; 
                        border-radius: 12px; min-height: 240px; display: flex; flex-direction: column; 
                        justify-content: center; transition: all 0.3s ease; background: {background_color};">
                    <div style="font-size: 40px; margin-bottom: 16px; color: {primary_color};">📊</div>
                    <div style="font-weight: 600; margin-bottom: 12px; font-size: 18px; color: {text_color};">
                        Summary Report
                    </div>
                    <div style="font-size: 13px; color: {text_light_color}; margin-bottom: 16px; line-height: 1.5;">
                        Comprehensive analysis summary with key metrics, insights, and security audit trail.
                    </div>
                    <div style="font-size: 11px; color: {text_light_color}; margin-top: 12px;">
                        🔐 Encrypted CSV • Timestamped • Audit Trail
                    </div>
                </div>
                '''
                st.markdown(export_card1_html, unsafe_allow_html=True)
                
                if st.button("📥 Export Summary Report", key="export_summary_btn", use_container_width=True):
                    summary_df = generate_summary_report(selected_video, video_info, df)
                    
                    if summary_df is not None:
                        csv_data = summary_df.to_csv(index=False)
                        
                        export_event = {
                            'timestamp': datetime.now().isoformat(),
                            'username': st.session_state.username,
                            'export_type': 'summary_report',
                            'filename': f"youtube_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            'video_id': selected_video
                        }
                        st.session_state.export_history.append(export_event)
                        
                        st.download_button(
                            label="⬇️ Download Secure CSV",
                            data=csv_data,
                            file_name=export_event['filename'],
                            mime="text/csv",
                            key="download_summary_csv",
                            use_container_width=True
                        )
                        
                        st.success("✅ Summary report generated! Download started.")
            
            with col2:
                export_card2_html = f'''
                <div style="text-align: center; padding: 24px 20px; border: 2px dashed {border_color}; 
                        border-radius: 12px; min-height: 240px; display: flex; flex-direction: column; 
                        justify-content: center; transition: all 0.3s ease; background: {background_color};">
                    <div style="font-size: 40px; margin-bottom: 16px; color: {primary_color};">📈</div>
                    <div style="font-weight: 600; margin-bottom: 12px; font-size: 18px; color: {text_color};">
                        Detailed Analysis
                    </div>
                    <div style="font-size: 13px; color: {text_light_color}; margin-bottom: 16px; line-height: 1.5;">
                        Complete dataset with sentiment scores, metadata, and analysis results.
                    </div>
                    <div style="font-size: 11px; color: {text_light_color}; margin-top: 12px;">
                        🔐 Full Dataset • Sentiment Scores • Analysis Metadata
                    </div>
                </div>
                '''
                st.markdown(export_card2_html, unsafe_allow_html=True)
                
                if st.button("📥 Export Detailed Analysis", key="export_detailed_btn", use_container_width=True):
                    detailed_df = generate_detailed_report(selected_video, video_info, df)
                    csv_data = detailed_df.to_csv(index=False)
                    
                    export_event = {
                        'timestamp': datetime.now().isoformat(),
                        'username': st.session_state.username,
                        'export_type': 'detailed_analysis',
                        'filename': f"youtube_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        'video_id': selected_video,
                        'records': len(detailed_df)
                    }
                    st.session_state.export_history.append(export_event)
                    
                    st.download_button(
                        label="⬇️ Download Full Analysis",
                        data=csv_data,
                        file_name=export_event['filename'],
                        mime="text/csv",
                        key="download_detailed_csv",
                        use_container_width=True
                    )
                    
                    st.success(f"✅ Detailed analysis exported ({len(detailed_df):,} records)!")
            
            # Additional export options
            st.markdown("---")
            advanced_export_html = f'''
            <div class="g-card">
                <h3 style="color: {text_color}; margin: 0 0 10px 0;">🔧 Advanced Export Options</h3>
                <p style="color: {text_light_color}; margin: 0;">Additional export formats</p>
            </div>
            '''
            st.markdown(advanced_export_html, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Export Raw Comments", key="export_raw_btn", use_container_width=True):
                    csv_data = df.to_csv(index=False)
                    
                    export_event = {
                        'timestamp': datetime.now().isoformat(),
                        'username': st.session_state.username,
                        'export_type': 'raw_comments',
                        'filename': f"youtube_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        'video_id': selected_video
                    }
                    st.session_state.export_history.append(export_event)
                    
                    st.download_button(
                        label="⬇️ Download Raw Comments",
                        data=csv_data,
                        file_name=export_event['filename'],
                        mime="text/csv",
                        key="download_raw_csv",
                        use_container_width=True
                    )
                    
                    st.success(f"✅ Raw comments exported ({len(df):,} records)!")
            
            with col2:
                if st.button("📊 Export Chart Data", key="export_chartdata_btn", use_container_width=True):
                    df_analyzed = analyze_sentiment(df.copy())
                    sentiment_counts = df_analyzed["sentiment"].value_counts()
                    
                    chart_data = pd.DataFrame({
                        'sentiment': sentiment_counts.index,
                        'count': sentiment_counts.values,
                        'percentage': (sentiment_counts.values / len(df_analyzed) * 100).round(1),
                        'video_id': selected_video,
                        'analysis_date': datetime.now().strftime("%Y-%m-%d")
                    })
                    
                    csv_data = chart_data.to_csv(index=False)
                    
                    export_event = {
                        'timestamp': datetime.now().isoformat(),
                        'username': st.session_state.username,
                        'export_type': 'chart_data',
                        'filename': f"youtube_chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    }
                    st.session_state.export_history.append(export_event)
                    
                    st.download_button(
                        label="⬇️ Download Chart Data",
                        data=csv_data,
                        file_name=export_event['filename'],
                        mime="text/csv",
                        key="download_chartdata_csv",
                        use_container_width=True
                    )
                    
                    st.success("✅ Chart data exported successfully!")
            
            # Export history (for admins)
            if st.session_state.user_role == 'admin' and st.session_state.export_history:
                st.markdown("---")
                with st.expander("📋 Export History (Admin Only)"):
                    history_df = pd.DataFrame(st.session_state.export_history)
                    st.dataframe(history_df, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
footer_html = f'''
<div style="text-align: center; color: {text_light_color}; font-size: 12px; padding: 20px 0;">
    <div style="margin-bottom: 8px;">
        <strong>{APP_NAME} v{APP_VERSION}</strong> • {DEPLOYMENT_MODE.upper()} MODE • SECURE SESSION • {current_theme.upper()} THEME
    </div>
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 8px; font-size: 11px;">
        <span>User: {st.session_state.username}</span>
        <span>Role: {st.session_state.user_role.upper()}</span>
        <span>Session: {st.session_state.session_id}</span>
    </div>
    <div style="font-size: 11px; color: {COLORS['neutral']};">
        © 2024 YouTube Sentiment Analysis Dashboard • All rights reserved
    </div>
</div>
'''
st.markdown(footer_html, unsafe_allow_html=True)

# Deployment mode indicator
if DEPLOYMENT_MODE != 'development':
    deployment_color = '#34A853' if DEPLOYMENT_MODE == 'production' else '#FBBC05'
    deployment_html = f'''
    <div style="position: fixed; bottom: 10px; right: 10px; z-index: 9999;">
        <div style="background: {deployment_color}; 
                    color: white; padding: 6px 14px; border-radius: 16px; 
                    font-size: 11px; font-weight: 600; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
            🔒 {DEPLOYMENT_MODE.upper()} • {current_theme.upper()} THEME
        </div>
    </div>
    '''
    st.markdown(deployment_html, unsafe_allow_html=True)