"""
Route optimization module using AI/ML algorithms
"""
import numpy as np
import networkx as nx
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouteOptimizer:
    def __init__(self):
        self.graph = nx.Graph()
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def create_station_network(self, stations_df):
        """Create a network graph of charging stations"""
        self.graph.clear()
        
        # Add nodes (charging stations)
        for idx, station in stations_df.iterrows():
            self.graph.add_node(
                station['station_id'],
                pos=(station['latitude'], station['longitude']),
                **station.to_dict()
            )
        
        # Add edges based on distance
        stations_list = list(self.graph.nodes())
        for i, station1 in enumerate(stations_list):
            for station2 in stations_list[i+1:]:
                pos1 = self.graph.nodes[station1]['pos']
                pos2 = self.graph.nodes[station2]['pos']
                distance = geodesic(pos1, pos2).kilometers
                
                # Only connect stations within reasonable distance
                if distance <= 100:  # 100 km threshold
                    self.graph.add_edge(station1, station2, weight=distance)
        
        logger.info(f"Created network with {len(self.graph.nodes())} stations and {len(self.graph.edges())} connections")
    
    def find_nearest_stations(self, user_lat, user_lon, stations_df, n_stations=10):
        """Find nearest charging stations using ML-enhanced scoring"""
        user_location = (user_lat, user_lon)
        
        # Calculate distances and features
        station_scores = []
        for idx, station in stations_df.iterrows():
            station_location = (station['latitude'], station['longitude'])
            distance = geodesic(user_location, station_location).kilometers
            
            # Calculate composite score
            availability_weight = 0.4
            distance_weight = 0.3
            rating_weight = 0.2
            power_weight = 0.1
            
            availability_score = station.get('availability_score', 0.5)
            distance_score = max(0, 1 - distance / 50)  # Normalize to 50km
            rating_score = station.get('rating', 4.0) / 5.0
            power_score = min(1.0, station.get('power_kw', 50) / 250)
            
            composite_score = (
                availability_score * availability_weight +
                distance_score * distance_weight +
                rating_score * rating_weight +
                power_score * power_weight
            )
            
            station_scores.append({
                'station_id': station['station_id'],
                'distance': distance,
                'score': composite_score,
                **station.to_dict()
            })
        
        # Sort by composite score
        station_scores.sort(key=lambda x: x['score'], reverse=True)
        return station_scores[:n_stations]
    
    def optimize_route_dijkstra(self, start_station, end_stations):
        """Optimize route using Dijkstra's algorithm"""
        if start_station not in self.graph.nodes():
            return None
        
        best_route = None
        best_distance = float('inf')
        
        for end_station in end_stations:
            if end_station not in self.graph.nodes():
                continue
                
            try:
                path = nx.shortest_path(
                    self.graph, 
                    start_station, 
                    end_station, 
                    weight='weight'
                )
                distance = nx.shortest_path_length(
                    self.graph, 
                    start_station, 
                    end_station, 
                    weight='weight'
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_route = {
                        'path': path,
                        'distance': distance,
                        'stations': [self.graph.nodes[station] for station in path]
                    }
            except nx.NetworkXNoPath:
                continue
        
        return best_route
    
    def optimize_route_astar(self, start_pos, end_pos, stations_df):
        """A* algorithm implementation for route optimization"""
        def heuristic(pos1, pos2):
            return geodesic(pos1, pos2).kilometers
        
        # Find nearest stations to start and end positions
        start_stations = self.find_nearest_stations(start_pos[0], start_pos[1], stations_df, 3)
        end_stations = self.find_nearest_stations(end_pos[0], end_pos[1], stations_df, 3)
        
        best_route = None
        best_total_distance = float('inf')
        
        for start_station in start_stations:
            for end_station in end_stations:
                start_id = start_station['station_id']
                end_id = end_station['station_id']
                
                if start_id not in self.graph.nodes() or end_id not in self.graph.nodes():
                    continue
                
                try:
                    # Use A* algorithm
                    path = nx.astar_path(
                        self.graph,
                        start_id,
                        end_id,
                        heuristic=lambda n1, n2: heuristic(
                            self.graph.nodes[n1]['pos'],
                            self.graph.nodes[n2]['pos']
                        ),
                        weight='weight'
                    )
                    
                    # Calculate total distance including start and end segments
                    start_distance = start_station['distance']
                    route_distance = nx.astar_path_length(
                        self.graph,
                        start_id,
                        end_id,
                        heuristic=lambda n1, n2: heuristic(
                            self.graph.nodes[n1]['pos'],
                            self.graph.nodes[n2]['pos']
                        ),
                        weight='weight'
                    )
                    end_distance = geodesic(
                        self.graph.nodes[end_id]['pos'],
                        end_pos
                    ).kilometers
                    
                    total_distance = start_distance + route_distance + end_distance
                    
                    if total_distance < best_total_distance:
                        best_total_distance = total_distance
                        best_route = {
                            'path': path,
                            'total_distance': total_distance,
                            'route_distance': route_distance,
                            'stations': [self.graph.nodes[station] for station in path],
                            'start_station': start_station,
                            'end_station': end_station
                        }
                        
                except nx.NetworkXNoPath:
                    continue
        
        return best_route
    
    def train_ml_model(self, stations_df, user_preferences):
        """Train ML model for personalized recommendations"""
        # Prepare training data
        features = []
        targets = []
        
        for idx, station in stations_df.iterrows():
            feature_vector = [
                station.get('distance_km', 0),
                station.get('power_kw', 50),
                station.get('rating', 4.0),
                station.get('availability_score', 0.5),
                station.get('price_per_kwh', 10),
                1 if station.get('amenities') != 'None' else 0
            ]
            
            # Create target based on composite scoring
            target = station.get('overall_score', 0.5)
            
            features.append(feature_vector)
            targets.append(target)
        
        # Train the model
        X = np.array(features)
        y = np.array(targets)
        
        self.ml_model.fit(X, y)
        self.is_trained = True
        
        logger.info("ML model trained successfully")
    
    def predict_station_preference(self, station_features):
        """Predict user preference for a station using trained ML model"""
        if not self.is_trained:
            return 0.5  # Default score
        
        features = np.array(station_features).reshape(1, -1)
        return self.ml_model.predict(features)[0]
    
    def cluster_stations(self, stations_df, n_clusters=5):
        """Cluster stations for better route planning"""
        coordinates = stations_df[['latitude', 'longitude']].values
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(coordinates)
        
        stations_df['cluster'] = clusters
        
        return stations_df, kmeans.cluster_centers_