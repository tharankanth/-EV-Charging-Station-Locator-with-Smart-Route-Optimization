"""
Main Streamlit application for EV Charging Station Locator
"""
import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import plotly.express as px
import time
import json

# Import custom modules
from data_processor import EVDataProcessor
from route_optimizer import RouteOptimizer
from chatbot_engine import EVChatbot
from visualization import EVVisualization
from config import *

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #f1f8e9;
        margin-right: 2rem;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1e5f7a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = EVDataProcessor()
if 'route_optimizer' not in st.session_state:
    st.session_state.route_optimizer = RouteOptimizer()
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = EVChatbot()
if 'visualization' not in st.session_state:
    st.session_state.visualization = EVVisualization()
if 'stations_df' not in st.session_state:
    st.session_state.stations_df = pd.DataFrame()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_location' not in st.session_state:
    st.session_state.user_location = None

@st.cache_data
def load_data():
    """Load and cache EV stations data"""
    data_processor = EVDataProcessor()
    stations_df = data_processor.load_ev_stations_data()
    chatbot_df = data_processor.load_chatbot_data()
    return stations_df, chatbot_df

@st.cache_data
def geocode_location(location_name):
    """Geocode location name to coordinates"""
    try:
        geolocator = Nominatim(user_agent="ev_locator")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        return None
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔋 EV Charging Station Locator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Smart Route Optimization for Electric Vehicles</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading EV charging stations data..."):
        stations_df, chatbot_df = load_data()
        st.session_state.stations_df = stations_df
    
    # Train chatbot if not already trained
    if not st.session_state.chatbot.is_trained:
        with st.spinner("Training AI chatbot..."):
            st.session_state.chatbot.train_chatbot(chatbot_df)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.selectbox(
            "Choose a feature:",
            ["🗺️ Station Locator", "🤖 AI Assistant", "📊 Analytics", "🛣️ Route Optimizer"]
        )
        
        st.header("📍 Location Settings")
        location_input = st.text_input("Enter your location:", placeholder="e.g., New Delhi, India")
        
        if st.button("📍 Set Location"):
            if location_input:
                coords = geocode_location(location_input)
                if coords:
                    st.session_state.user_location = coords
                    st.success(f"Location set to: {location_input}")
                else:
                    st.error("Could not find the location. Please try again.")
        
        if st.session_state.user_location:
            st.info(f"📍 Current location: {st.session_state.user_location[0]:.4f}, {st.session_state.user_location[1]:.4f}")
        
        # Filters
        st.header("🔧 Filters")
        max_distance = st.slider("Maximum distance (km)", 1, 100, 50)
        min_power = st.slider("Minimum power (kW)", 7, 250, 50)
        connector_types = st.multiselect(
            "Connector types:",
            ["Type 2", "CCS", "CHAdeMO", "Tesla"],
            default=["Type 2", "CCS"]
        )
        availability_filter = st.multiselect(
            "Availability status:",
            ["Available", "Occupied", "Maintenance"],
            default=["Available"]
        )
    
    # Main content based on selected page
    if page == "🗺️ Station Locator":
        show_station_locator(stations_df, max_distance, min_power, connector_types, availability_filter)
    elif page == "🤖 AI Assistant":
        show_ai_assistant()
    elif page == "📊 Analytics":
        show_analytics(stations_df)
    elif page == "🛣️ Route Optimizer":
        show_route_optimizer(stations_df)

def show_station_locator(stations_df, max_distance, min_power, connector_types, availability_filter):
    """Display station locator interface"""
    st.header("🗺️ Find Charging Stations")
    
    if st.session_state.user_location is None:
        st.warning("Please set your location in the sidebar to find nearby stations.")
        return
    
    # Process data with user location
    user_lat, user_lon = st.session_state.user_location
    processed_df = st.session_state.data_processor.calculate_distance_features(
        stations_df.copy(), user_lat, user_lon
    )
    
    # Apply filters
    filtered_df = processed_df[
        (processed_df['distance_km'] <= max_distance) &
        (processed_df['power_kw'] >= min_power) &
        (processed_df['connector_type'].isin(connector_types)) &
        (processed_df['availability'].isin(availability_filter))
    ].copy()
    
    if filtered_df.empty:
        st.warning("No stations found matching your criteria. Try adjusting the filters.")
        return
    
    # Sort by distance
    filtered_df = filtered_df.sort_values('distance_km')
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏪 Total Stations", len(filtered_df))
    with col2:
        available_count = len(filtered_df[filtered_df['availability'] == 'Available'])
        st.metric("🟢 Available", available_count)
    with col3:
        avg_distance = filtered_df['distance_km'].mean()
        st.metric("📏 Avg Distance", f"{avg_distance:.1f} km")
    with col4:
        fast_charging = len(filtered_df[filtered_df['power_kw'] >= 50])
        st.metric("⚡ Fast Charging", fast_charging)
    
    # Create and display map
    st.subheader("📍 Interactive Map")
    station_map = st.session_state.visualization.create_station_map(
        filtered_df, st.session_state.user_location
    )
    
    if station_map:
        map_data = st_folium(station_map, width=700, height=500)
    
    # Display station list
    st.subheader("📋 Nearby Stations")
    
    # Show top 10 nearest stations
    top_stations = filtered_df.head(10)
    
    for idx, station in top_stations.iterrows():
        with st.expander(f"🏪 {station['name']} - {station['distance_km']:.1f} km"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**⚡ Power:** {station['power_kw']} kW")
                st.write(f"**🔌 Connector:** {station['connector_type']}")
                st.write(f"**🟢 Status:** {station['availability']}")
            
            with col2:
                st.write(f"**💰 Price:** ₹{station['price_per_kwh']:.1f}/kWh")
                st.write(f"**⭐ Rating:** {station['rating']:.1f}/5")
                st.write(f"**🏪 Amenities:** {station.get('amenities', 'None')}")
            
            if st.button(f"🛣️ Get Directions", key=f"directions_{idx}"):
                st.info(f"Directions to {station['name']} would open in your default map application.")

def show_ai_assistant():
    """Display AI chatbot interface"""
    st.header("🤖 AI Charging Assistant")
    st.write("Ask me anything about EV charging stations, routes, pricing, or technical questions!")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">🤖 **Assistant:** {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Type your message:", key="chat_input", placeholder="e.g., Find fast charging stations near me")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("Send 📤")
    with col2:
        clear_button = st.button("Clear 🗑️")
    
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Prepare context
        context = {}
        if st.session_state.user_location and not st.session_state.stations_df.empty:
            user_lat, user_lon = st.session_state.user_location
            processed_df = st.session_state.data_processor.calculate_distance_features(
                st.session_state.stations_df.copy(), user_lat, user_lon
            )
            nearest_stations = processed_df.nsmallest(5, 'distance_km').to_dict('records')
            context['nearest_stations'] = nearest_stations
            context['station_count'] = len(processed_df)
        
        # Get bot response
        with st.spinner("🤖 Thinking..."):
            bot_response = st.session_state.chatbot.get_response(user_input, context)
        
        # Add bot response to history
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        # Rerun to update the display
        st.rerun()
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Quick action buttons
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Find Stations"):
            st.session_state.chat_history.append({"role": "user", "content": "Find charging stations near me"})
            st.rerun()
    
    with col2:
        if st.button("💰 Check Pricing"):
            st.session_state.chat_history.append({"role": "user", "content": "What are the charging costs?"})
            st.rerun()
    
    with col3:
        if st.button("⚡ Fast Charging"):
            st.session_state.chat_history.append({"role": "user", "content": "Tell me about fast charging options"})
            st.rerun()

def show_analytics(stations_df):
    """Display analytics dashboard"""
    st.header("📊 EV Charging Analytics")
    
    if stations_df.empty:
        st.warning("No data available for analytics.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏪 Total Stations", len(stations_df))
    with col2:
        avg_power = stations_df['power_kw'].mean()
        st.metric("⚡ Avg Power", f"{avg_power:.0f} kW")
    with col3:
        avg_rating = stations_df['rating'].mean()
        st.metric("⭐ Avg Rating", f"{avg_rating:.1f}/5")
    with col4:
        avg_price = stations_df['price_per_kwh'].mean()
        st.metric("💰 Avg Price", f"₹{avg_price:.1f}/kWh")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Availability chart
        availability_chart = st.session_state.visualization.create_availability_chart(stations_df)
        if availability_chart:
            st.plotly_chart(availability_chart, use_container_width=True)
    
    with col2:
        # Power distribution
        power_chart = st.session_state.visualization.create_power_distribution_chart(stations_df)
        if power_chart:
            st.plotly_chart(power_chart, use_container_width=True)
    
    # Price comparison
    st.subheader("💰 Price Analysis")
    price_chart = st.session_state.visualization.create_price_comparison_chart(stations_df)
    if price_chart:
        st.plotly_chart(price_chart, use_container_width=True)
    
    # City-wise analysis
    if 'city' in stations_df.columns:
        st.subheader("🏙️ City-wise Distribution")
        city_stats = stations_df.groupby('city').agg({
            'station_id': 'count',
            'power_kw': 'mean',
            'rating': 'mean',
            'price_per_kwh': 'mean'
        }).round(2)
        city_stats.columns = ['Station Count', 'Avg Power (kW)', 'Avg Rating', 'Avg Price (₹/kWh)']
        st.dataframe(city_stats, use_container_width=True)
    
    # Detailed analytics
    with st.expander("📈 Detailed Analytics"):
        # Correlation analysis
        numeric_cols = ['power_kw', 'rating', 'price_per_kwh', 'availability_score']
        available_cols = [col for col in numeric_cols if col in stations_df.columns]
        
        if len(available_cols) > 1:
            correlation_matrix = stations_df[available_cols].corr()
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_route_optimizer(stations_df):
    """Display route optimization interface"""
    st.header("🛣️ Smart Route Optimizer")
    
    if st.session_state.user_location is None:
        st.warning("Please set your location in the sidebar to use route optimization.")
        return
    
    # Destination input
    destination = st.text_input("Enter destination:", placeholder="e.g., Mumbai, India")
    
    col1, col2 = st.columns(2)
    with col1:
        vehicle_range = st.number_input("Vehicle range (km):", min_value=100, max_value=500, value=300)
    with col2:
        current_charge = st.slider("Current battery level (%):", 0, 100, 80)
    
    if st.button("🔍 Optimize Route") and destination:
        dest_coords = geocode_location(destination)
        
        if dest_coords is None:
            st.error("Could not find the destination. Please try again.")
            return
        
        with st.spinner("🧠 Optimizing your route..."):
            # Calculate distance features
            user_lat, user_lon = st.session_state.user_location
            processed_df = st.session_state.data_processor.calculate_distance_features(
                stations_df.copy(), user_lat, user_lon
            )
            
            # Create network and find optimal route
            st.session_state.route_optimizer.create_station_network(processed_df)
            
            # Find route using A* algorithm
            optimal_route = st.session_state.route_optimizer.optimize_route_astar(
                st.session_state.user_location, dest_coords, processed_df
            )
            
            if optimal_route:
                st.success("✅ Route optimized successfully!")
                
                # Display route information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📏 Total Distance", f"{optimal_route['total_distance']:.1f} km")
                with col2:
                    estimated_time = optimal_route['total_distance'] / 60 * 60  # Rough estimate
                    st.metric("⏱️ Estimated Time", f"{estimated_time:.0f} min")
                with col3:
                    charging_stops = len(optimal_route['stations'])
                    st.metric("🔋 Charging Stops", charging_stops)
                
                # Display route map
                st.subheader("🗺️ Optimized Route")
                route_map = st.session_state.visualization.create_station_map(
                    processed_df, st.session_state.user_location, optimal_route
                )
                
                if route_map:
                    # Add destination marker
                    folium.Marker(
                        location=dest_coords,
                        popup="Destination",
                        tooltip="Your destination",
                        icon=folium.Icon(color='red', icon='flag', prefix='fa')
                    ).add_to(route_map)
                    
                    st_folium(route_map, width=700, height=500)
                
                # Display charging stops
                st.subheader("🔋 Recommended Charging Stops")
                for i, station in enumerate(optimal_route['stations'], 1):
                    with st.expander(f"Stop {i}: {station['name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**📍 Location:** {station['latitude']:.4f}, {station['longitude']:.4f}")
                            st.write(f"**⚡ Power:** {station['power_kw']} kW")
                            st.write(f"**🔌 Connector:** {station['connector_type']}")
                        with col2:
                            st.write(f"**🟢 Status:** {station['availability']}")
                            st.write(f"**💰 Price:** ₹{station['price_per_kwh']:.1f}/kWh")
                            st.write(f"**⭐ Rating:** {station['rating']:.1f}/5")
            else:
                st.error("Could not find an optimal route. Please try different parameters.")
    
    # Route planning tips
    with st.expander("💡 Route Planning Tips"):
        st.markdown("""
        **🎯 Optimization Tips:**
        - Plan charging stops when battery is around 20-30%
        - Fast charging is most efficient between 20-80% battery level
        - Consider amenities for longer charging sessions
        - Check real-time availability before starting your journey
        - Keep alternative stations in mind for backup
        
        **⚡ Charging Strategy:**
        - Use fast chargers for quick top-ups during travel
        - Slow charging overnight at destinations
        - Plan charging during meal breaks or shopping
        """)

if __name__ == "__main__":
    main()