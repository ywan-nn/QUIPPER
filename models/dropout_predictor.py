import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DropoutPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = [
            'days_active', 'total_logins', 'avg_session_min', 
            'num_quizzes', 'avg_quiz_score', 'progress_rate',
            'forum_posts', 'assignment_submissions'
        ]
        
    def prepare_data(self, df):
        """Prepare features and target"""
        # Create target: High risk = 1, else 0
        df['high_risk'] = (df['risk_category'] == 'High').astype(int)
        
        X = df[self.features]
        y = df['high_risk']
        
        return X, y
    
    def train(self, df):
        """Train the model"""
        X, y = self.prepare_data(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        feature_importance = dict(zip(self.features, self.model.feature_importances_))
        
        # Save model
        os.makedirs('models/saved', exist_ok=True)
        joblib.dump(self.model, 'models/saved/dropout_model.pkl')
        joblib.dump(self.scaler, 'models/saved/scaler.pkl')
        
        return {
            'accuracy': self.model.score(X_test_scaled, y_test),
            'feature_importance': feature_importance
        }
    
    def predict(self, df):
        """Make predictions"""
        if self.model is None:
            # Load saved model
            self.model = joblib.load('models/saved/dropout_model.pkl')
            self.scaler = joblib.load('models/saved/scaler.pkl')
        
        X = df[self.features]
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)
        
        return predictions, probabilities
    
    def get_risk_factors(self, student_data):
        """Get top risk factors for a student"""
        if self.model is None:
            self.model = joblib.load('models/saved/dropout_model.pkl')
            
        # Find out why student is at risk
        # Simple heuristic based on feature thresholds
        risk_factors = []
        
        if student_data['days_active'] < 14:
            risk_factors.append(f"Low activity: only {student_data['days_active']:.0f} active days")
        if student_data['avg_quiz_score'] < 40:
            risk_factors.append(f"Poor quiz performance: {student_data['avg_quiz_score']:.1f}% average")
        if student_data['progress_rate'] < 30:
            risk_factors.append(f"Low progress: only {student_data['progress_rate']:.1f}% completed")
        if student_data['forum_posts'] == 0 and student_data['assignment_submissions'] < 2:
            risk_factors.append("Low engagement: no forum posts, few assignments")
            
        return risk_factors

if __name__ == "__main__":
    # Test the model
    df = pd.read_csv('data/student_data.csv')
    predictor = DropoutPredictor()
    results = predictor.train(df)
    print("Training Results:", results)