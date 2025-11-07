"""
AI Chatbot engine for EV charging assistance
"""
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = MultinomialNB()
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.classifier)
        ])
        self.responses = {}
        self.intents = {}
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Tokenize and remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            filtered_text = [w for w in word_tokens if w not in stop_words]
            return ' '.join(filtered_text)
        except:
            return text
    
    def load_training_data(self, chatbot_df):
        """Load and prepare training data"""
        self.training_data = chatbot_df.copy()
        
        # Add EV-specific intents and responses
        ev_specific_data = [
            ("find charging station", "location", "I can help you find nearby charging stations. Please share your current location or city."),
            ("charging cost price", "pricing", "Charging costs typically range from ₹8-15 per kWh depending on the station and location."),
            ("fast charging available", "charging_speed", "Yes! We have fast charging stations ranging from 50kW to 250kW. Fast charging can charge your EV to 80% in 30-45 minutes."),
            ("connector type support", "connectors", "We support all major connector types: Type 2, CCS (Combined Charging System), CHAdeMO, and Tesla connectors."),
            ("station availability check", "availability", "You can check real-time availability on our interactive map. Green markers indicate available stations."),
            ("booking reservation", "booking", "Yes, you can reserve charging slots through our app up to 24 hours in advance."),
            ("payment methods accepted", "payment", "We accept credit cards, debit cards, UPI, and popular mobile wallets like Paytm, PhonePe, and Google Pay."),
            ("charging time duration", "charging_time", "Charging time varies by your vehicle's battery capacity and charger power. Typically: Slow (7kW): 6-8 hours, Fast (50kW): 1-2 hours, Ultra-fast (150kW+): 20-45 minutes."),
            ("membership plans benefits", "membership", "Our membership plans offer discounted rates, priority booking, and exclusive access to premium stations."),
            ("report issue problem", "support", "You can report issues through our app, website, or call our 24/7 customer support. We'll help you find alternative stations immediately."),
            ("station amenities facilities", "amenities", "Many stations offer parking, restrooms, restaurants, shopping centers, and WiFi while you charge."),
            ("route planning optimization", "route", "Our AI-powered route planner finds the optimal path with charging stops based on your vehicle's range and preferences."),
            ("vehicle compatibility", "compatibility", "Our system works with all electric vehicles. Just select your vehicle model for optimized charging recommendations."),
            ("green energy renewable", "sustainability", "Many of our partner stations use renewable energy sources like solar and wind power for eco-friendly charging."),
            ("emergency assistance help", "emergency", "For emergencies, call our 24/7 helpline. We provide roadside assistance and emergency charging solutions.")
        ]
        
        # Add to training data
        for query, intent, response in ev_specific_data:
            new_row = pd.DataFrame({
                'input': [query],
                'response': [response],
                'intent': [intent]
            })
            self.training_data = pd.concat([self.training_data, new_row], ignore_index=True)
        
        logger.info(f"Loaded {len(self.training_data)} training examples")
    
    def train_chatbot(self, chatbot_df):
        """Train the chatbot model"""
        self.load_training_data(chatbot_df)
        
        # Preprocess training data
        processed_inputs = [self.preprocess_text(text) for text in self.training_data['input']]
        
        # Create intent labels if not present
        if 'intent' not in self.training_data.columns:
            self.training_data['intent'] = 'general'
        
        # Train the classifier
        self.pipeline.fit(processed_inputs, self.training_data['intent'])
        
        # Store responses by intent
        for idx, row in self.training_data.iterrows():
            intent = row['intent']
            if intent not in self.responses:
                self.responses[intent] = []
            self.responses[intent].append(row['response'])
        
        self.is_trained = True
        logger.info("Chatbot training completed")
    
    def get_response(self, user_input, context=None):
        """Generate response for user input"""
        if not self.is_trained:
            return "I'm still learning. Please train me first!"
        
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        # Predict intent
        try:
            predicted_intent = self.pipeline.predict([processed_input])[0]
            confidence = max(self.pipeline.predict_proba([processed_input])[0])
            
            # If confidence is too low, use similarity matching
            if confidence < 0.3:
                return self._similarity_based_response(user_input)
            
            # Get response for predicted intent
            if predicted_intent in self.responses:
                responses = self.responses[predicted_intent]
                # Select best response (for now, random selection)
                response = np.random.choice(responses)
                
                # Add context-aware information if available
                if context:
                    response = self._add_context_info(response, context, predicted_intent)
                
                return response
            else:
                return self._similarity_based_response(user_input)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble understanding. Could you please rephrase your question?"
    
    def _similarity_based_response(self, user_input):
        """Fallback to similarity-based matching"""
        processed_input = self.preprocess_text(user_input)
        
        # Calculate similarity with all training inputs
        training_inputs = [self.preprocess_text(text) for text in self.training_data['input']]
        
        # Vectorize
        all_texts = training_inputs + [processed_input]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Calculate similarity
        similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
        
        # Find best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > 0.1:  # Minimum similarity threshold
            return self.training_data.iloc[best_match_idx]['response']
        else:
            return self._get_default_response()
    
    def _add_context_info(self, response, context, intent):
        """Add contextual information to response"""
        if intent == 'location' and 'nearest_stations' in context:
            stations = context['nearest_stations'][:3]  # Top 3 stations
            station_info = "\n\nNearest stations:\n"
            for i, station in enumerate(stations, 1):
                station_info += f"{i}. {station['name']} - {station['distance']:.1f}km away\n"
            response += station_info
        
        elif intent == 'availability' and 'station_count' in context:
            response += f"\n\nCurrently showing {context['station_count']} stations in your area."
        
        return response
    
    def _get_default_response(self):
        """Get default response when no good match is found"""
        default_responses = [
            "I'm here to help with EV charging questions! You can ask me about finding stations, pricing, charging times, or booking slots.",
            "I can help you with EV charging information. Try asking about nearby stations, connector types, or charging costs.",
            "I'm your EV charging assistant! Ask me about station locations, availability, or any charging-related questions.",
            "I can assist with finding charging stations, checking availability, or answering questions about EV charging. What would you like to know?"
        ]
        return np.random.choice(default_responses)
    
    def get_intent(self, user_input):
        """Get the predicted intent for user input"""
        if not self.is_trained:
            return "unknown"
        
        processed_input = self.preprocess_text(user_input)
        try:
            return self.pipeline.predict([processed_input])[0]
        except:
            return "unknown"
    
    def get_confidence(self, user_input):
        """Get confidence score for the prediction"""
        if not self.is_trained:
            return 0.0
        
        processed_input = self.preprocess_text(user_input)
        try:
            return max(self.pipeline.predict_proba([processed_input])[0])
        except:
            return 0.0