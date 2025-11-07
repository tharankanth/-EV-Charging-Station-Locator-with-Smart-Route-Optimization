"""
Configuration settings for EV Charging Station Locator
"""
import os

# API Keys (replace with your actual keys)
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'your_google_maps_api_key')
OPENROUTE_API_KEY = os.getenv('OPENROUTE_API_KEY', 'your_openroute_api_key')

# Default coordinates (New Delhi, India)
DEFAULT_LAT = 28.6139
DEFAULT_LON = 77.2090

# Map settings
MAP_ZOOM = 12
SEARCH_RADIUS = 50  # km

# Model settings
MODEL_PATH = 'models/'
CHATBOT_MODEL_PATH = 'models/chatbot_model.pkl'
ROUTE_MODEL_PATH = 'models/route_optimizer.pkl'

# Dataset paths
EV_STATIONS_DATA = 'data/ev_charging_stations.csv'
CHATBOT_DATA = 'data/chatbot_dialogs.csv'

# Streamlit settings
PAGE_TITLE = "EV Charging Station Locator"
PAGE_ICON = "🔋"
LAYOUT = "wide"

# Colors for visualization
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8'
}