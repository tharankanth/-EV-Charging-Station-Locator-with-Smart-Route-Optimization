"""
Machine Learning model training for EV charging station recommendations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression()
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def prepare_features(self, stations_df, user_preferences=None):
        """Prepare features for model training"""
        features_df = stations_df.copy()
        
        # Handle missing values
        features_df = features_df.fillna({
            'power_kw': features_df['power_kw'].median(),
            'rating': features_df['rating'].median(),
            'price_per_kwh': features_df['price_per_kwh'].median(),
            'availability_score': 0.5
        })
        
        # Encode categorical variables
        categorical_cols = ['connector_type', 'availability', 'amenities', 'city']
        for col in categorical_cols:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create derived features
        features_df['power_category'] = pd.cut(
            features_df['power_kw'], 
            bins=[0, 22, 50, 150, 300], 
            labels=['Slow', 'Medium', 'Fast', 'Ultra-fast']
        )
        features_df['power_category_encoded'] = LabelEncoder().fit_transform(features_df['power_category'])
        
        # Distance-based features (if available)
        if 'distance_km' in features_df.columns:
            features_df['distance_category'] = pd.cut(
                features_df['distance_km'],
                bins=[0, 5, 15, 30, 100],
                labels=['Very Close', 'Close', 'Medium', 'Far']
            )
            features_df['distance_score'] = 1 / (1 + features_df['distance_km'] / 10)
        
        # Price category
        features_df['price_category'] = pd.cut(
            features_df['price_per_kwh'],
            bins=[0, 8, 12, 15, 20],
            labels=['Cheap', 'Moderate', 'Expensive', 'Very Expensive']
        )
        features_df['price_category_encoded'] = LabelEncoder().fit_transform(features_df['price_category'])
        
        # Create target variable (composite score)
        features_df['preference_score'] = self._calculate_preference_score(features_df, user_preferences)
        
        return features_df
    
    def _calculate_preference_score(self, df, user_preferences=None):
        """Calculate preference score based on multiple factors"""
        # Default weights
        weights = {
            'distance': 0.3,
            'availability': 0.25,
            'rating': 0.2,
            'power': 0.15,
            'price': 0.1
        }
        
        # Update weights based on user preferences
        if user_preferences:
            weights.update(user_preferences)
        
        # Normalize scores to 0-1 range
        distance_score = 1 - (df.get('distance_km', 0) / df.get('distance_km', 0).max()) if 'distance_km' in df.columns else 0.5
        availability_score = df.get('availability_score', 0.5)
        rating_score = df.get('rating', 4.0) / 5.0
        power_score = df.get('power_kw', 50) / df.get('power_kw', 50).max()
        price_score = 1 - (df.get('price_per_kwh', 10) / df.get('price_per_kwh', 10).max())
        
        # Calculate weighted composite score
        composite_score = (
            distance_score * weights['distance'] +
            availability_score * weights['availability'] +
            rating_score * weights['rating'] +
            power_score * weights['power'] +
            price_score * weights['price']
        )
        
        return composite_score
    
    def select_features(self, features_df):
        """Select relevant features for training"""
        feature_columns = [
            'power_kw', 'rating', 'price_per_kwh', 'availability_score',
            'connector_type_encoded', 'availability_encoded', 'amenities_encoded',
            'power_category_encoded', 'price_category_encoded'
        ]
        
        # Add distance features if available
        if 'distance_km' in features_df.columns:
            feature_columns.extend(['distance_km', 'distance_score'])
        
        # Add city encoding if available
        if 'city_encoded' in features_df.columns:
            feature_columns.append('city_encoded')
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        return features_df[available_features], features_df['preference_score']
    
    def train_models(self, stations_df, user_preferences=None):
        """Train multiple models and select the best one"""
        logger.info("Starting model training...")
        
        # Prepare features
        features_df = self.prepare_features(stations_df, user_preferences)
        X, y = self.select_features(features_df)
        
        if X.empty or len(X) < 10:
            logger.error("Insufficient data for training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate models
        model_scores = {}
        trained_models = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                if name == 'linear_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'model': model
                }
                
                trained_models[name] = model
                
                logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Select best model based on R² score
        if model_scores:
            best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['r2'])
            self.best_model = trained_models[best_model_name]
            
            logger.info(f"Best model: {best_model_name} with R² = {model_scores[best_model_name]['r2']:.4f}")
            
            # Get feature importance for tree-based models
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("Top 5 important features:")
                for idx, row in self.feature_importance.head().iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return model_scores
        
        return None
    
    def hyperparameter_tuning(self, X, y, model_name='random_forest'):
        """Perform hyperparameter tuning for the specified model"""
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def cross_validate_model(self, X, y, model=None, cv=5):
        """Perform cross-validation"""
        if model is None:
            model = self.best_model
        
        if model is None:
            logger.error("No model available for cross-validation")
            return None
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        logger.info(f"Cross-validation R² scores: {scores}")
        logger.info(f"Mean R² score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, filepath='models/ev_recommendation_model.pkl'):
        """Save the trained model"""
        if self.best_model is None:
            logger.error("No trained model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and preprocessing objects
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_importance': self.feature_importance
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath='models/ev_recommendation_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_importance = model_data.get('feature_importance')
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_preferences(self, stations_df, user_location=None):
        """Predict user preferences for stations"""
        if self.best_model is None:
            logger.error("No trained model available for prediction")
            return None
        
        try:
            # Prepare features
            features_df = self.prepare_features(stations_df)
            X, _ = self.select_features(features_df)
            
            # Make predictions
            if hasattr(self.best_model, 'predict'):
                predictions = self.best_model.predict(X)
                
                # Add predictions to dataframe
                result_df = stations_df.copy()
                result_df['predicted_preference'] = predictions
                result_df = result_df.sort_values('predicted_preference', ascending=False)
                
                return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def cluster_stations(self, stations_df, n_clusters=5):
        """Cluster stations for better recommendations"""
        try:
            # Select features for clustering
            cluster_features = ['latitude', 'longitude', 'power_kw', 'rating', 'price_per_kwh']
            available_features = [col for col in cluster_features if col in stations_df.columns]
            
            if len(available_features) < 2:
                logger.error("Insufficient features for clustering")
                return None
            
            X = stations_df[available_features].fillna(stations_df[available_features].median())
            
            # Standardize features
            X_scaled = StandardScaler().fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to dataframe
            result_df = stations_df.copy()
            result_df['cluster'] = clusters
            
            logger.info(f"Stations clustered into {n_clusters} groups")
            
            return result_df, kmeans
            
        except Exception as e:
            logger.error(f"Error clustering stations: {e}")
            return None, None

def main():
    """Main function for training models"""
    from data_processor import EVDataProcessor
    
    # Initialize components
    data_processor = EVDataProcessor()
    model_trainer = EVModelTrainer()
    
    # Load data
    logger.info("Loading data...")
    stations_df = data_processor.load_ev_stations_data()
    
    if stations_df.empty:
        logger.error("No data available for training")
        return
    
    # Add synthetic user location for training
    user_lat, user_lon = 28.6139, 77.2090  # Delhi coordinates
    stations_df = data_processor.calculate_distance_features(stations_df, user_lat, user_lon)
    
    # Train models
    model_scores = model_trainer.train_models(stations_df)
    
    if model_scores:
        # Save the best model
        model_trainer.save_model()
        
        # Perform clustering
        clustered_df, kmeans_model = model_trainer.cluster_stations(stations_df)
        if clustered_df is not None:
            logger.info("Station clustering completed")
        
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed")

if __name__ == "__main__":
    main()