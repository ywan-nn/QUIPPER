import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_student_data(num_students=200):
    """Generate sample student data for Quipper"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Base student profiles
    students = []
    schools = ['SMA Negeri 1 Jakarta', 'SMA Bina Bangsa', 'SMA Al-Azhar', 
               'SMA Tunas Muda', 'SMA Cendekia', 'SMA Islam Terpadu']
    cities = ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar', 'Yogyakarta']
    courses = ['Matematika', 'Fisika', 'Kimia', 'Biologi', 'Bahasa Inggris']
    
    for i in range(num_students):
        # Student profile
        student = {
            'student_id': f'STU{str(i+1).zfill(4)}',
            'name': f'Siswa {i+1}',
            'school': random.choice(schools),
            'city': random.choice(cities),
            'course': random.choice(courses),
            'enrollment_date': datetime.now() - timedelta(days=random.randint(30, 180)),
        }
        
        # Learning behavior
        days_active = random.randint(1, 90)
        logins = np.random.poisson(2.5, days_active).sum() if days_active > 0 else 0
        avg_session_min = random.uniform(5, 60)
        
        # Quiz performance
        num_quizzes = random.randint(3, 15)
        quiz_scores = np.random.beta(2, 2, num_quizzes) * 100
        avg_quiz_score = quiz_scores.mean()
        
        # Course progress
        modules_completed = random.randint(0, 20)
        total_modules = random.randint(20, 30)
        progress_rate = min(100, (modules_completed / total_modules) * 100)
        
        # Engagement metrics
        forum_posts = np.random.poisson(0.5)
        assignment_submissions = random.randint(0, 8)
        
        # Determine dropout risk based on behavior patterns
        risk_score = 0
        
        # Low activity = high risk
        if days_active < 14:
            risk_score += 30
        elif days_active < 30:
            risk_score += 15
            
        # Low quiz scores = high risk  
        if avg_quiz_score < 40:
            risk_score += 25
        elif avg_quiz_score < 60:
            risk_score += 10
            
        # Low progress = high risk
        if progress_rate < 30:
            risk_score += 25
        elif progress_rate < 50:
            risk_score += 10
            
        # Low engagement = high risk
        if forum_posts == 0 and assignment_submissions < 2:
            risk_score += 20
        elif forum_posts == 0:
            risk_score += 10
            
        # Random noise
        risk_score += random.randint(-10, 10)
        risk_score = max(0, min(100, risk_score))
        
        # Determine risk category
        if risk_score >= 70:
            risk_category = 'High'
            dropout_probability = random.uniform(0.7, 0.95)
        elif risk_score >= 40:
            risk_category = 'Medium'
            dropout_probability = random.uniform(0.3, 0.7)
        else:
            risk_category = 'Low'
            dropout_probability = random.uniform(0.05, 0.3)
        
        # Generate sentiment data (feedback text)
        feedback_templates = {
            'positive': [
                "Materi sangat membantu!",
                "Penyampaiannya jelas dan mudah dipahami.",
                "Latihan soalnya bagus untuk persiapan ujian.",
                "Platformnya user friendly dan interaktif.",
                "Saya jadi lebih paham setelah mengikuti kelas ini."
            ],
            'negative': [
                "Terlalu cepat, saya ketinggalan.",
                "Penjelasannya kurang detail.",
                "Latihan soalnya terlalu sulit.",
                "Sering buffering, mengganggu fokus.",
                "Materinya kurang menarik."
            ],
            'neutral': [
                "Belum ada perkembangan, masih sama.",
                "Kurang feedback dari mentor.",
                "Materi sudah sesuai dengan kurikulum.",
                "Tugasnya cukup banyak."
            ]
        }
        
        if risk_category == 'High':
            sentiment = random.choices(['negative', 'neutral', 'positive'], weights=[0.5, 0.3, 0.2])[0]
        elif risk_category == 'Medium':
            sentiment = random.choices(['neutral', 'positive', 'negative'], weights=[0.4, 0.4, 0.2])[0]
        else:
            sentiment = random.choices(['positive', 'neutral', 'negative'], weights=[0.5, 0.3, 0.2])[0]
            
        feedback_text = random.choice(feedback_templates[sentiment])
        
        # Compile student data
        student.update({
            'days_active': days_active,
            'total_logins': logins,
            'avg_session_min': round(avg_session_min, 1),
            'num_quizzes': num_quizzes,
            'avg_quiz_score': round(avg_quiz_score, 1),
            'progress_rate': round(progress_rate, 1),
            'modules_completed': modules_completed,
            'total_modules': total_modules,
            'forum_posts': forum_posts,
            'assignment_submissions': assignment_submissions,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'dropout_probability': round(dropout_probability, 3),
            'sentiment': sentiment,
            'sentiment_score': 1 if sentiment == 'positive' else (-1 if sentiment == 'negative' else 0),
            'feedback_text': feedback_text,
            'last_activity': datetime.now() - timedelta(days=random.randint(0, 14))
        })
        
        students.append(student)
    
    return pd.DataFrame(students)

if __name__ == "__main__":
    df = generate_student_data(200)
    df.to_csv('data/student_data.csv', index=False)
    print(f"Generated {len(df)} student records")
    print(df.head())