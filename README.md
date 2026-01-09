# youtube-sentiment-apps
# YouTube Sentiment Analysis Dashboard

A Streamlit app for analyzing sentiment in YouTube comments.

## Features
- Fetch and analyze YouTube comments
- Sentiment analysis using TextBlob
- Interactive charts with Plotly
- PDF report generation
- Multi-video comparison

## Deployment on Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file path to `app.py`
6. Add YouTube API key in secrets

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
