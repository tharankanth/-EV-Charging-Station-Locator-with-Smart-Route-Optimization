"""
Utility functions for EV Charging Station Locator
"""
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta
import json
import logging
import requests
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationUtils:
    """Utilities for location-based operations"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        try:
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    @staticmethod
    def find_stations_in_radius(stations_df: pd.DataFrame, center_lat: float, 
                               center_lon: float, radius_km: float) -> pd.DataFrame:
        """Find all stations within a given radius"""
        if stations_df.empty:
            return pd.DataFrame()
        
        distances = []
        for _, station in stations_df.iterrows():
            distance = LocationUtils.calculate_distance(
                center_lat, center_lon, station['latitude'], station['longitude']
            )
            distances.append(distance)
        
        stations_df = stations_df.copy()
        stations_df['distance_km'] = distances
        
        return stations_df[stations_df['distance_km'] <= radius_km].sort_values('distance_km')
    
    @staticmethod
    def get_bounding_box(lat: float, lon: float, radius_km: float) -> Dict[str, float]:
        """Get bounding box coordinates for a given center and radius"""
        # Approximate conversion: 1 degree ≈ 111 km
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * np.cos(np.radians(lat)))
        
        return {
            'north': lat + lat_delta,
            'south': lat - lat_delta,
            'east': lon + lon_delta,
            'west': lon - lon_delta
        }

class DataValidation:
    """Data validation utilities"""
    
    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """Validate latitude and longitude values"""
        return -90 <= lat <= 90 and -180 <= lon <= 180
    
    @staticmethod
    def validate_station_data(station: Dict) -> bool:
        """Validate station data completeness"""
        required_fields = ['station_id', 'name', 'latitude', 'longitude']
        return all(field in station and station[field] is not None for field in required_fields)
    
    @staticmethod
    def clean_station_data(stations_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate station data"""
        if stations_df.empty:
            return stations_df
        
        # Remove rows with invalid coordinates
        valid_coords = stations_df.apply(
            lambda row: DataValidation.validate_coordinates(row['latitude'], row['longitude']), 
            axis=1
        )
        stations_df = stations_df[valid_coords]
        
        # Fill missing values
        stations_df = stations_df.fillna({
            'power_kw': 50,
            'rating': 4.0,
            'price_per_kwh': 10.0,
            'availability': 'Unknown',
            'connector_type': 'Type 2',
            'amenities': 'None'
        })
        
        # Ensure data types
        stations_df['power_kw'] = pd.to_numeric(stations_df['power_kw'], errors='coerce').fillna(50)
        stations_df['rating'] = pd.to_numeric(stations_df['rating'], errors='coerce').fillna(4.0)
        stations_df['price_per_kwh'] = pd.to_numeric(stations_df['price_per_kwh'], errors='coerce').fillna(10.0)
        
        return stations_df

class ChargingCalculator:
    """Utilities for charging calculations"""
    
    @staticmethod
    def estimate_charging_time(battery_capacity_kwh: float, current_soc: float, 
                              target_soc: float, charging_power_kw: float) -> float:
        """Estimate charging time in hours"""
        try:
            energy_needed = battery_capacity_kwh * (target_soc - current_soc) / 100
            
            # Apply charging curve (efficiency decreases at higher SoC)
            if target_soc > 80:
                effective_power = charging_power_kw * 0.6  # Reduced power at high SoC
            elif target_soc > 60:
                effective_power = charging_power_kw * 0.8
            else:
                effective_power = charging_power_kw
            
            charging_time = energy_needed / effective_power
            return max(0, charging_time)
            
        except Exception as e:
            logger.error(f"Error calculating charging time: {e}")
            return 0.0
    
    @staticmethod
    def calculate_charging_cost(energy_kwh: float, price_per_kwh: float, 
                               membership_discount: float = 0.0) -> float:
        """Calculate charging cost"""
        try:
            base_cost = energy_kwh * price_per_kwh
            discount_amount = base_cost * membership_discount / 100
            return max(0, base_cost - discount_amount)
        except Exception as e:
            logger.error(f"Error calculating charging cost: {e}")
            return 0.0
    
    @staticmethod
    def estimate_range_after_charging(current_range_km: float, energy_added_kwh: float, 
                                    vehicle_efficiency_kwh_per_100km: float) -> float:
        """Estimate vehicle range after charging"""
        try:
            additional_range = (energy_added_kwh / vehicle_efficiency_kwh_per_100km) * 100
            return current_range_km + additional_range
        except Exception as e:
            logger.error(f"Error calculating range: {e}")
            return current_range_km

class RouteUtils:
    """Utilities for route planning"""
    
    @staticmethod
    def calculate_route_distance(waypoints: List[Tuple[float, float]]) -> float:
        """Calculate total distance for a route with waypoints"""
        if len(waypoints) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            distance = LocationUtils.calculate_distance(
                waypoints[i][0], waypoints[i][1],
                waypoints[i+1][0], waypoints[i+1][1]
            )
            total_distance += distance
        
        return total_distance
    
    @staticmethod
    def find_intermediate_stations(start: Tuple[float, float], end: Tuple[float, float], 
                                 stations_df: pd.DataFrame, max_detour_km: float = 20) -> pd.DataFrame:
        """Find stations along a route within acceptable detour distance"""
        if stations_df.empty:
            return pd.DataFrame()
        
        direct_distance = LocationUtils.calculate_distance(start[0], start[1], end[0], end[1])
        
        suitable_stations = []
        for _, station in stations_df.iterrows():
            station_pos = (station['latitude'], station['longitude'])
            
            # Calculate distance from start to station to end
            dist_start_station = LocationUtils.calculate_distance(start[0], start[1], station_pos[0], station_pos[1])
            dist_station_end = LocationUtils.calculate_distance(station_pos[0], station_pos[1], end[0], end[1])
            
            total_with_station = dist_start_station + dist_station_end
            detour = total_with_station - direct_distance
            
            if detour <= max_detour_km:
                station_data = station.to_dict()
                station_data['detour_km'] = detour
                station_data['distance_from_start'] = dist_start_station
                suitable_stations.append(station_data)
        
        if suitable_stations:
            return pd.DataFrame(suitable_stations).sort_values('detour_km')
        else:
            return pd.DataFrame()

class TimeUtils:
    """Time-related utilities"""
    
    @staticmethod
    def get_time_category(hour: int) -> str:
        """Categorize time of day"""
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Night"
    
    @staticmethod
    def estimate_arrival_time(distance_km: float, avg_speed_kmh: float = 60) -> datetime:
        """Estimate arrival time based on distance and speed"""
        travel_time_hours = distance_km / avg_speed_kmh
        return datetime.now() + timedelta(hours=travel_time_hours)
    
    @staticmethod
    def format_duration(hours: float) -> str:
        """Format duration in a human-readable format"""
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} min"
        elif hours < 24:
            hours_int = int(hours)
            minutes = int((hours - hours_int) * 60)
            return f"{hours_int}h {minutes}m"
        else:
            days = int(hours / 24)
            remaining_hours = int(hours % 24)
            return f"{days}d {remaining_hours}h"

class APIUtils:
    """API-related utilities"""
    
    @staticmethod
    def make_safe_request(url: str, params: Dict = None, timeout: int = 10) -> Optional[Dict]:
        """Make a safe HTTP request with error handling"""
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
    
    @staticmethod
    def geocode_address(address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to coordinates (mock implementation)"""
        # This is a simplified implementation
        # In a real application, you would use a geocoding service
        city_coordinates = {
            'delhi': (28.6139, 77.2090),
            'mumbai': (19.0760, 72.8777),
            'bangalore': (12.9716, 77.5946),
            'chennai': (13.0827, 80.2707),
            'kolkata': (22.5726, 88.3639),
            'hyderabad': (17.3850, 78.4867),
            'pune': (18.5204, 73.8567),
            'ahmedabad': (23.0225, 72.5714)
        }
        
        address_lower = address.lower()
        for city, coords in city_coordinates.items():
            if city in address_lower:
                return coords
        
        return None

class ExportUtils:
    """Data export utilities"""
    
    @staticmethod
    def export_stations_to_csv(stations_df: pd.DataFrame, filename: str) -> bool:
        """Export stations data to CSV"""
        try:
            stations_df.to_csv(filename, index=False)
            logger.info(f"Data exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    @staticmethod
    def export_route_to_json(route_data: Dict, filename: str) -> bool:
        """Export route data to JSON"""
        try:
            with open(filename, 'w') as f:
                json.dump(route_data, f, indent=2, default=str)
            logger.info(f"Route exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting route: {e}")
            return False

class PerformanceUtils:
    """Performance optimization utilities"""
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert to category if few unique values
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':
                # Downcast float64 to float32 if possible
                if df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
                    df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                # Downcast int64 to smaller int types if possible
                if df[col].min() > np.iinfo(np.int32).min and df[col].max() < np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
        
        return df
    
    @staticmethod
    def batch_process_stations(stations_df: pd.DataFrame, batch_size: int = 1000) -> List[pd.DataFrame]:
        """Split large DataFrame into batches for processing"""
        batches = []
        for i in range(0, len(stations_df), batch_size):
            batch = stations_df.iloc[i:i+batch_size].copy()
            batches.append(batch)
        return batches

# Convenience functions
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points"""
    return LocationUtils.calculate_distance(lat1, lon1, lat2, lon2)

def validate_station(station: Dict) -> bool:
    """Validate a single station record"""
    return DataValidation.validate_station_data(station)

def estimate_charging_duration(battery_kwh: float, current_percent: float, 
                             target_percent: float, power_kw: float) -> float:
    """Estimate charging duration in hours"""
    return ChargingCalculator.estimate_charging_time(battery_kwh, current_percent, target_percent, power_kw)