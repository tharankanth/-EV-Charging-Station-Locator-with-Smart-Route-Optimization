"""
Data processing module for EV charging station data
"""
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_ev_stations_data(self, file_path=None):
        """Load and preprocess EV charging stations data"""
        try:
            if file_path:
                df = pd.read_csv(file_path)
            else:
                # Create synthetic data for demonstration
                df = self._create_synthetic_ev_data()
            
            logger.info(f"Loaded {len(df)} EV charging stations")
            return self.preprocess_ev_data(df)
        except Exception as e:
            logger.error(f"Error loading EV data: {e}")
            return self._create_synthetic_ev_data()
    
    def _create_synthetic_ev_data(self):
        """Create synthetic EV charging station data"""
        np.random.seed(42)
        n_stations = 500
        
        # Generate coordinates around major Indian cities
        cities = [
            (28.6139, 77.2090, "Delhi"),
            (19.0760, 72.8777, "Mumbai"),
            (12.9716, 77.5946, "Bangalore"),
            (13.0827, 80.2707, "Chennai"),
            (22.5726, 88.3639, "Kolkata")
        ]
        
        data = []
        for i in range(n_stations):
            city = cities[i % len(cities)]
            lat = city[0] + np.random.normal(0, 0.1)
            lon = city[1] + np.random.normal(0, 0.1)
            
            station = {
                'station_id': f'EV_{i+1:03d}',
                'name': f'Charging Station {i+1}',
                'latitude': lat,
                'longitude': lon,
                'city': city[2],
                'connector_type': np.random.choice(['Type 2', 'CCS', 'CHAdeMO', 'Tesla']),
                'power_kw': np.random.choice([7, 22, 50, 150, 250]),
                'availability': np.random.choice(['Available', 'Occupied', 'Maintenance'], p=[0.7, 0.2, 0.1]),
                'price_per_kwh': np.random.uniform(8, 15),
                'rating': np.random.uniform(3.5, 5.0),
                'amenities': np.random.choice(['Parking', 'Restaurant', 'Shopping', 'None'], p=[0.3, 0.2, 0.2, 0.3])
            }
            data.append(station)
        
        return pd.DataFrame(data)
    
    def preprocess_ev_data(self, df):
        """Preprocess EV charging station data"""
        # Handle missing values
        df = df.fillna({
            'availability': 'Unknown',
            'price_per_kwh': df['price_per_kwh'].median() if 'price_per_kwh' in df.columns else 10,
            'rating': df['rating'].median() if 'rating' in df.columns else 4.0
        })
        
        # Create additional features
        df['is_fast_charging'] = df['power_kw'] >= 50
        df['availability_score'] = df['availability'].map({
            'Available': 1.0,
            'Occupied': 0.3,
            'Maintenance': 0.0,
            'Unknown': 0.5
        })
        
        # Normalize coordinates for clustering
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['lat_norm'] = self.scaler.fit_transform(df[['latitude']])
            df['lon_norm'] = self.scaler.fit_transform(df[['longitude']])
        
        return df
    
    def calculate_distance_features(self, df, user_lat, user_lon):
        """Calculate distance-based features"""
        user_location = (user_lat, user_lon)
        
        distances = []
        for _, row in df.iterrows():
            station_location = (row['latitude'], row['longitude'])
            distance = geodesic(user_location, station_location).kilometers
            distances.append(distance)
        
        df['distance_km'] = distances
        df['distance_score'] = 1 / (1 + df['distance_km'] / 10)  # Inverse distance scoring
        
        return df
    
    def create_station_features(self, df):
        """Create features for ML model"""
        features = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['connector_type', 'availability', 'amenities']
        for col in categorical_cols:
            if col in features.columns:
                features[f'{col}_encoded'] = self.label_encoder.fit_transform(features[col].astype(str))
        
        # Create composite scores
        features['overall_score'] = (
            features['rating'] * 0.3 +
            features['availability_score'] * 0.4 +
            features['distance_score'] * 0.3
        )
        
        return features
    
    def load_chatbot_data(self, file_path=None):
        """Load and preprocess chatbot training data"""
        try:
            if file_path:
                df = pd.read_csv(file_path)
            else:
                df = self._create_synthetic_chatbot_data()
            
            logger.info(f"Loaded {len(df)} chatbot dialog pairs")
            return df
        except Exception as e:
            logger.error(f"Error loading chatbot data: {e}")
            return self._create_synthetic_chatbot_data()
    
    def _create_synthetic_chatbot_data(self):
        """Create synthetic chatbot training data"""
        dialogs = [
            ("Hello", "Hi! How can I help you with EV charging today?"),
            ("Find charging stations near me", "I can help you find nearby charging stations. Please share your location."),
            ("What types of connectors are available?", "We support Type 2, CCS, CHAdeMO, and Tesla connectors."),
            ("How much does charging cost?", "Charging costs vary by location, typically between ₹8-15 per kWh."),
            ("Is fast charging available?", "Yes, we have fast charging stations with 50kW to 250kW power."),
            ("What amenities are available?", "Many stations offer parking, restaurants, and shopping facilities."),
            ("How do I check availability?", "You can check real-time availability on our map interface."),
            ("What is the charging time?", "Charging time depends on your vehicle and charger power. Fast chargers can charge 80% in 30-45 minutes."),
            ("Do you have 24/7 stations?", "Yes, many of our partner stations operate 24/7."),
            ("How do I report issues?", "You can report issues through our app or contact customer support."),
            ("What payment methods do you accept?", "We accept credit cards, debit cards, and mobile wallets."),
            ("Can I reserve a charging slot?", "Yes, you can reserve charging slots through our mobile app."),
            ("What if the charger is not working?", "Please report the issue immediately. We'll help you find alternative stations."),
            ("Do you have membership plans?", "Yes, we offer various membership plans with discounted rates."),
            ("How accurate is the availability status?", "Our availability status is updated in real-time with 95% accuracy.")
        ]
        
        return pd.DataFrame(dialogs, columns=['input', 'response'])