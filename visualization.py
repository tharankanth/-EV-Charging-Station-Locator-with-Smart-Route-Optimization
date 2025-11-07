"""
Visualization module for EV charging station maps and analytics
"""
import folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class EVVisualization:
    def __init__(self):
        self.color_map = {
            'Available': 'green',
            'Occupied': 'orange',
            'Maintenance': 'red',
            'Unknown': 'gray'
        }
        
    def create_station_map(self, stations_df, user_location=None, selected_route=None):
        """Create interactive map with charging stations"""
        if stations_df.empty:
            return None
        
        # Calculate map center
        if user_location:
            center_lat, center_lon = user_location
        else:
            center_lat = stations_df['latitude'].mean()
            center_lon = stations_df['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add user location marker
        if user_location:
            folium.Marker(
                location=user_location,
                popup="Your Location",
                tooltip="You are here",
                icon=folium.Icon(color='blue', icon='user', prefix='fa')
            ).add_to(m)
        
        # Add charging station markers
        for idx, station in stations_df.iterrows():
            # Determine marker color based on availability
            color = self.color_map.get(station.get('availability', 'Unknown'), 'gray')
            
            # Create popup content
            popup_content = f"""
            <div style="width: 200px;">
                <h4>{station['name']}</h4>
                <p><b>Distance:</b> {station.get('distance_km', 0):.1f} km</p>
                <p><b>Power:</b> {station.get('power_kw', 'N/A')} kW</p>
                <p><b>Connector:</b> {station.get('connector_type', 'N/A')}</p>
                <p><b>Availability:</b> {station.get('availability', 'Unknown')}</p>
                <p><b>Price:</b> ₹{station.get('price_per_kwh', 0):.1f}/kWh</p>
                <p><b>Rating:</b> {station.get('rating', 0):.1f}/5</p>
            </div>
            """
            
            # Determine icon based on power
            if station.get('power_kw', 0) >= 150:
                icon = 'bolt'
            elif station.get('power_kw', 0) >= 50:
                icon = 'flash'
            else:
                icon = 'plug'
            
            folium.Marker(
                location=[station['latitude'], station['longitude']],
                popup=folium.Popup(popup_content, max_width=250),
                tooltip=f"{station['name']} - {station.get('availability', 'Unknown')}",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)
        
        # Add route if provided
        if selected_route and 'stations' in selected_route:
            route_coords = []
            if user_location:
                route_coords.append(user_location)
            
            for station in selected_route['stations']:
                route_coords.append([station['latitude'], station['longitude']])
            
            folium.PolyLine(
                locations=route_coords,
                color='blue',
                weight=4,
                opacity=0.8,
                popup=f"Optimized Route - {selected_route.get('total_distance', 0):.1f} km"
            ).add_to(m)
        
        return m
    
    def create_availability_chart(self, stations_df):
        """Create availability status chart"""
        if stations_df.empty:
            return None
        
        availability_counts = stations_df['availability'].value_counts()
        
        fig = px.pie(
            values=availability_counts.values,
            names=availability_counts.index,
            title="Charging Station Availability Status",
            color_discrete_map=self.color_map
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def create_power_distribution_chart(self, stations_df):
        """Create power distribution histogram"""
        if stations_df.empty:
            return None
        
        fig = px.histogram(
            stations_df,
            x='power_kw',
            nbins=20,
            title="Distribution of Charging Power (kW)",
            labels={'power_kw': 'Power (kW)', 'count': 'Number of Stations'}
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_distance_vs_rating_scatter(self, stations_df):
        """Create scatter plot of distance vs rating"""
        if stations_df.empty or 'distance_km' not in stations_df.columns:
            return None
        
        fig = px.scatter(
            stations_df,
            x='distance_km',
            y='rating',
            size='power_kw',
            color='availability',
            hover_data=['name', 'price_per_kwh'],
            title="Distance vs Rating (Size = Power)",
            labels={'distance_km': 'Distance (km)', 'rating': 'Rating (1-5)'},
            color_discrete_map=self.color_map
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_price_comparison_chart(self, stations_df):
        """Create price comparison chart"""
        if stations_df.empty:
            return None
        
        # Group by connector type and calculate average price
        price_by_connector = stations_df.groupby('connector_type')['price_per_kwh'].agg(['mean', 'min', 'max']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Average Price',
            x=price_by_connector['connector_type'],
            y=price_by_connector['mean'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=price_by_connector['max'] - price_by_connector['mean'],
                arrayminus=price_by_connector['mean'] - price_by_connector['min']
            )
        ))
        
        fig.update_layout(
            title="Average Charging Price by Connector Type",
            xaxis_title="Connector Type",
            yaxis_title="Price (₹/kWh)",
            height=400
        )
        
        return fig
    
    def create_analytics_dashboard(self, stations_df):
        """Create comprehensive analytics dashboard"""
        if stations_df.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Station Count by City', 'Power Distribution', 
                          'Availability Status', 'Rating Distribution'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Station count by city
        if 'city' in stations_df.columns:
            city_counts = stations_df['city'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=city_counts.index, y=city_counts.values, name="Stations"),
                row=1, col=1
            )
        
        # Power distribution
        fig.add_trace(
            go.Histogram(x=stations_df['power_kw'], name="Power", nbinsx=15),
            row=1, col=2
        )
        
        # Availability status
        availability_counts = stations_df['availability'].value_counts()
        fig.add_trace(
            go.Pie(labels=availability_counts.index, values=availability_counts.values, name="Availability"),
            row=2, col=1
        )
        
        # Rating distribution
        fig.add_trace(
            go.Histogram(x=stations_df['rating'], name="Rating", nbinsx=10),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="EV Charging Stations Analytics Dashboard")
        
        return fig
    
    def create_route_optimization_viz(self, route_data):
        """Visualize route optimization results"""
        if not route_data:
            return None
        
        # Create comparison chart of different routes
        routes = ['Current Route', 'Optimized Route']
        distances = [route_data.get('original_distance', 0), route_data.get('total_distance', 0)]
        times = [route_data.get('original_time', 0), route_data.get('estimated_time', 0)]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distance Comparison', 'Time Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=routes, y=distances, name="Distance (km)", marker_color=['red', 'green']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=routes, y=times, name="Time (min)", marker_color=['red', 'green']),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="Route Optimization Results")
        
        return fig
    
    def create_station_details_card(self, station):
        """Create detailed station information card"""
        return f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h3 style="color: #2E86AB; margin-top: 0;">{station['name']}</h3>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <p><strong>📍 Distance:</strong> {station.get('distance_km', 0):.1f} km</p>
                    <p><strong>⚡ Power:</strong> {station.get('power_kw', 'N/A')} kW</p>
                    <p><strong>🔌 Connector:</strong> {station.get('connector_type', 'N/A')}</p>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <p><strong>🟢 Status:</strong> {station.get('availability', 'Unknown')}</p>
                    <p><strong>💰 Price:</strong> ₹{station.get('price_per_kwh', 0):.1f}/kWh</p>
                    <p><strong>⭐ Rating:</strong> {station.get('rating', 0):.1f}/5</p>
                </div>
            </div>
            {f"<p><strong>🏪 Amenities:</strong> {station.get('amenities', 'None')}</p>" if station.get('amenities') != 'None' else ""}
        </div>
        """