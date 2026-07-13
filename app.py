import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.dropout_predictor import DropoutPredictor
from models.sentiment_analyzer import SentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="AILA - AI Learning Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - PROFESIONAL dengan Font Awesome
st.markdown("""
<style>
    /* Font Awesome for professional icons */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        color: #E31E24;
        margin-bottom: 0;
        font-weight: 700;
    }
    .main-header i {
        color: #E31E24;
        margin-right: 12px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E31E24 0%, #B71C1C 100%);
        padding-top: 20px;
        overflow-y: auto !important;
        max-height: 100vh !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]:hover {
        background-color: #f0f0f0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox span {
        color: #333333 !important;
    }
    [data-testid="stSidebar"] .stSelectbox div {
        color: #333333 !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255,255,255,0.9) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }
    [data-testid="stSidebar"] .sidebar-icon {
        color: white;
        margin-right: 10px;
        font-size: 1.1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #f8f9fa;
        padding: 15px 10px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e9ecef;
        border-bottom: 4px solid #E31E24;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(227, 30, 36, 0.15);
        transform: translateY(-2px);
    }
    .metric-card .metric-icon {
        font-size: 1.8rem;
        color: #E31E24;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #999;
        margin-top: 2px;
    }
    
    /* Risk Colors */
    .risk-high {
        color: #E31E24 !important;
    }
    .risk-medium {
        color: #FF6B00 !important;
    }
    .risk-low {
        color: #28a745 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #E31E24;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #B71C1C;
        color: white;
        transform: scale(1.02);
    }
    
    /* Feedback Cards */
    .feedback-positive {
        background-color: #d4edda;
        padding: 8px 12px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .feedback-negative {
        background-color: #f8d7da;
        padding: 8px 12px;
        border-radius: 8px;
        border-left: 4px solid #E31E24;
    }
    .feedback-neutral {
        background-color: #fff3cd;
        padding: 8px 12px;
        border-radius: 8px;
        border-left: 4px solid #FF6B00;
    }
    
    /* FAQ Styling */
    .faq-item {
        background: #f8f9fa;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 4px solid #E31E24;
        transition: all 0.3s ease;
    }
    .faq-item:hover {
        box-shadow: 0 2px 8px rgba(227, 30, 36, 0.15);
        transform: translateX(4px);
    }
    .faq-question {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1a1a1a;
        margin-bottom: 6px;
    }
    .faq-answer {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .faq-category {
        background: #E31E24;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
        background-color: #f0f0f0;
        color: #333;
    }
    .stTabs [data-baseweb="tab"] i {
        margin-right: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E31E24 !important;
        color: white !important;
    }
    .stTabs [aria-selected="true"] i {
        color: white !important;
    }
    
    /* Section headers with icons */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 16px;
    }
    .section-header i {
        color: #E31E24;
        margin-right: 10px;
    }
    
    /* Sidebar icon styling */
    .sidebar-filter-label i {
        margin-right: 8px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/student_data.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Generating sample data...")
        from data.generate_sample_data import generate_student_data
        df = generate_student_data(200)
        df.to_csv('data/student_data.csv', index=False)
        return df

def generate_intervention_message(student, risk_factors):
    name = student['name']
    course = student['course']
    progress = student['progress_rate']
    
    if progress < 20:
        message = f"Halo {name}! Saya lihat kamu masih di awal perjalanan belajar {course}. Yuk kita mulai dari yang paling dasar dulu!"
    elif len(risk_factors) > 0 and 'quiz' in str(risk_factors[0]).lower():
        message = f"Halo {name}! Materi {course} memang menantang, tapi kamu pasti bisa! Coba tonton video pembahasan soal-soal sebelumnya."
    elif len(risk_factors) > 0 and 'activity' in str(risk_factors[0]).lower():
        message = f"Halo {name}! {course} minggu ini seru banget lho, yuk masuk kelas lagi! Ada materi baru yang menarik."
    else:
        message = f"Halo {name}! Bagaimana perkembangan belajar {course} hari ini? Ada yang bisa saya bantu?"
    
    return message

def show_dashboard(filtered_df, dropout_predictor, sentiment_analyzer, total_students, high_risk, avg_progress, avg_quiz):
    """Menampilkan halaman Dashboard - Sesuai BRD FR-01, FR-02, FR-04"""
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<p class="main-header"><i class="fas fa-robot"></i> AILA - AI Learning Analytics</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p class="sub-header" style="text-align: right;"><i class="far fa-calendar-alt"></i> Quipper | {datetime.now().strftime("%d %B %Y")}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # FR-01: KPI Metrics
    st.markdown('<p class="section-header"><i class="fas fa-chart-pie"></i> Key Performance Indicators</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon"><i class="fas fa-users"></i></div>
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total_students}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon"><i class="fas fa-exclamation-triangle"></i></div>
            <div class="metric-label">High Risk Students</div>
            <div class="metric-value risk-high">{high_risk}</div>
            <div class="metric-sub">{high_risk/total_students*100:.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon"><i class="fas fa-chart-line"></i></div>
            <div class="metric-label">Average Progress</div>
            <div class="metric-value">{avg_progress:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon"><i class="fas fa-pencil-alt"></i></div>
            <div class="metric-label">Average Quiz Score</div>
            <div class="metric-value">{avg_quiz:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # FR-01 & FR-02: Risk Analysis
    st.markdown('<p class="section-header"><i class="fas fa-shield-alt"></i> Student Risk Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        risk_counts = filtered_df['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        colors = {'High': '#E74C3C', 'Medium': '#F39C12', 'Low': '#2ECC71'}
        fig1 = px.bar(risk_counts, x='Risk Level', y='Count', 
                     color='Risk Level',
                     color_discrete_map=colors,
                     title="Risk Level Distribution")
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        risk_by_course = filtered_df.groupby('course').agg({
            'risk_score': 'mean',
            'student_id': 'count'
        }).reset_index()
        risk_by_course.columns = ['Course', 'Avg Risk Score', 'Students']
        risk_by_course = risk_by_course.sort_values('Avg Risk Score', ascending=False)
        fig2 = px.bar(risk_by_course.head(8), x='Course', y='Avg Risk Score',
                     title="Average Risk Score by Course",
                     labels={'Course': 'Course', 'Avg Risk Score': 'Avg Risk Score'},
                     color='Avg Risk Score',
                     color_continuous_scale='Blues')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # FR-02: High Risk Students with Root Cause Analysis
    st.markdown('<p class="section-header"><i class="fas fa-list-ul"></i> High Risk Students - Priority Intervention</p>', unsafe_allow_html=True)
    
    high_risk_students = filtered_df[filtered_df['risk_category'] == 'High'].copy()
    high_risk_students = high_risk_students.sort_values('risk_score', ascending=False)
    
    if len(high_risk_students) > 0:
        risk_factors = []
        root_causes = []
        for _, student in high_risk_students.head(10).iterrows():
            factors = dropout_predictor.get_risk_factors(student)
            risk_factors.append(', '.join(factors) if factors else 'Multiple risk indicators')
            causes = []
            if student['days_active'] < 14:
                causes.append("Low activity")
            if student['avg_quiz_score'] < 40:
                causes.append("Poor quiz performance")
            if student['progress_rate'] < 30:
                causes.append("Low progress")
            if student['forum_posts'] == 0:
                causes.append("No engagement")
            root_causes.append(', '.join(causes) if causes else 'Multiple factors')
        
        display_df = high_risk_students.head(10)[['student_id', 'name', 'course', 'risk_score', 'progress_rate', 'avg_quiz_score', 'days_active']].copy()
        display_df['risk_factors'] = risk_factors[:len(display_df)]
        display_df['root_causes'] = root_causes[:len(display_df)]
        
        styled_df = display_df.style.map(
            lambda x: 'color: #E31E24; font-weight: bold;' if isinstance(x, (int, float)) and x > 70 else '',
            subset=['risk_score']
        )
        st.dataframe(styled_df, use_container_width=True)
        
        st.info("💡 **AI Intervention Suggestions:**")
        for _, student in high_risk_students.head(3).iterrows():
            factors = dropout_predictor.get_risk_factors(student)
            message = generate_intervention_message(student, factors)
            st.write(f"• **{student['name']}** (ID: {student['student_id']}): {message}")
    else:
        st.success("✅ No high risk students found! Great job!")
    
    st.markdown("---")
    
    # FR-04: Sentiment Analysis
    st.markdown('<p class="section-header"><i class="fas fa-comment-dots"></i> Student Sentiment Analysis</p>', unsafe_allow_html=True)
    
    if 'feedback_text' in filtered_df.columns and len(filtered_df) > 0:
        results = sentiment_analyzer.analyze_batch(filtered_df['feedback_text'].tolist())
        filtered_df['predicted_sentiment'] = [r['sentiment'] for r in results]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            sent_counts = filtered_df['predicted_sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Count']
            colors2 = {'positive': '#2ECC71', 'negative': '#E74C3C', 'neutral': '#F39C12'}
            fig3 = px.pie(sent_counts, values='Count', names='Sentiment', 
                         title='Sentiment Distribution',
                         color='Sentiment',
                         color_discrete_map=colors2)
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            sent_by_course = filtered_df.groupby(['course', 'predicted_sentiment']).size().reset_index(name='count')
            fig4 = px.bar(sent_by_course, x='course', y='count', color='predicted_sentiment',
                         title='Sentiment by Course',
                         color_discrete_map=colors2,
                         barmode='group')
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)
        
        with col3:
            negative_feedback = filtered_df[filtered_df['predicted_sentiment'] == 'negative']['feedback_text'].tolist()
            if negative_feedback:
                from collections import Counter
                all_words = ' '.join(negative_feedback).lower().split()
                common_issues = Counter([w for w in all_words if len(w) > 4]).most_common(5)
                st.write("**Top Issues from Negative Feedback:**")
                for word, count in common_issues:
                    st.write(f"• {word}: {count} mentions")
            else:
                st.write("✅ No negative feedback found!")
    
    st.markdown("---")
    
    # FR-04: Feedback Summary Report
    st.markdown('<p class="section-header"><i class="fas fa-file-alt"></i> Student Feedback Summary Report</p>', unsafe_allow_html=True)
    
    if 'feedback_text' in filtered_df.columns and len(filtered_df) > 0:
        summary = sentiment_analyzer.get_sentiment_summary(filtered_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon"><i class="fas fa-inbox"></i></div>
                <div style="font-size:0.9rem; color:#666;">Total Feedback</div>
                <div class="metric-value">{summary['total_analyzed']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_feedback = summary['total_analyzed']
            positive_pct = (summary['positive'] / total_feedback * 100) if total_feedback > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon" style="color:#2ECC71;"><i class="fas fa-thumbs-up"></i></div>
                <div style="font-size:0.9rem; color:#666;">Positive Feedback</div>
                <div class="metric-value" style="color:#2ECC71;">{summary['positive']}</div>
                <div style="font-size:0.8rem; color:#666;">{positive_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_feedback = summary['total_analyzed']
            negative_pct = (summary['negative'] / total_feedback * 100) if total_feedback > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon" style="color:#E74C3C;"><i class="fas fa-thumbs-down"></i></div>
                <div style="font-size:0.9rem; color:#666;">Negative Feedback</div>
                <div class="metric-value" style="color:#E74C3C;">{summary['negative']}</div>
                <div style="font-size:0.8rem; color:#666;">{negative_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        feedback_data = []
        for idx, row in filtered_df.iterrows():
            if pd.notna(row.get('feedback_text')):
                sentiment_result = sentiment_analyzer.analyze_feedback(row.get('feedback_text', ''))
                feedback_data.append({
                    'Student': row.get('name', 'Unknown'),
                    'Course': row.get('course', 'Unknown'),
                    'Feedback': row.get('feedback_text', ''),
                    'Sentiment': sentiment_result['sentiment'].upper(),
                    'Polarity': sentiment_result['polarity'],
                    'Keywords': ', '.join([kw['word'] for kw in sentiment_result['keywords_found'][:3]])
                })
        
        if feedback_data:
            feedback_df = pd.DataFrame(feedback_data)
            
            def color_sentiment(val):
                if val == 'POSITIVE':
                    return 'color: #2ECC71'
                elif val == 'NEGATIVE':
                    return 'color: #E74C3C'
                else:
                    return 'color: #F39C12'
            
            st.dataframe(
                feedback_df.style.map(color_sentiment, subset=['Sentiment']),
                use_container_width=True,
                height=300
            )
            
            csv = feedback_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feedback Summary (CSV)",
                data=csv,
                file_name=f"feedback_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        if 'top_keywords' in summary and summary['top_keywords']:
            st.markdown('<p class="section-header"><i class="fas fa-tags"></i> Top Keywords from Feedback</p>', unsafe_allow_html=True)
            cols = st.columns(5)
            for idx, (word, count) in enumerate(summary['top_keywords'][:5]):
                if idx < 5:
                    cols[idx].metric(label=word.capitalize(), value=count)
    else:
        st.info("No feedback data available for analysis")
    
    st.markdown("---")
    
    # Learning Progress
    st.markdown('<p class="section-header"><i class="fas fa-chart-bar"></i> Learning Progress Tracking</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(filtered_df) > 0:
            fig5 = px.histogram(
                filtered_df, 
                x='progress_rate', 
                title='Student Progress Distribution',
                labels={'progress_rate': 'Progress (%)', 'count': 'Count'},
                nbins=20,
                color_discrete_sequence=['#4A90D9']
            )
            fig5.update_layout(height=350)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No data to display")
    
    with col2:
        if len(filtered_df) > 0:
            fig6 = px.scatter(
                filtered_df, 
                x='progress_rate', 
                y='avg_quiz_score',
                color='risk_category',
                title='Quiz Score vs Progress',
                labels={'progress_rate': 'Progress (%)', 'avg_quiz_score': 'Avg Quiz Score'},
                color_discrete_map={'High': '#E74C3C', 'Medium': '#F39C12', 'Low': '#2ECC71'},
                hover_data=['student_id', 'name'],
                size='total_logins'
            )
            fig6.update_layout(height=350)
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No data to display")
    
    st.markdown("---")
    st.markdown('<p class="section-header"><i class="fas fa-lightbulb"></i> AI Learning Recommendations</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**<i class='fas fa-chart-simple'></i> Dashboard AI Insights:**", unsafe_allow_html=True)
        insights = []
        
        if total_students > 0 and high_risk > total_students * 0.2:
            insights.append(f"⚠️ {high_risk/total_students*100:.1f}% of students are at high risk. Immediate intervention recommended.")
        
        if total_students > 0:
            low_engagement = len(filtered_df[filtered_df['days_active'] < 7])
            if low_engagement > total_students * 0.1:
                insights.append(f"⚠️ {low_engagement} students have been inactive for over a week. Proactive engagement needed.")
            
            low_quiz = len(filtered_df[filtered_df['avg_quiz_score'] < 40])
            if low_quiz > total_students * 0.1:
                insights.append(f"⚠️ {low_quiz} students are struggling with quizzes. Additional support recommended.")
        
        if not insights:
            insights = ["✅ All metrics are in good standing. Continue monitoring."]
        
        for insight in insights:
            st.write(insight)
    
    with col2:
        st.markdown("**<i class='fas fa-graduation-cap'></i> Course Improvement Recommendations:**", unsafe_allow_html=True)
        if len(filtered_df) > 10:
            course_avg_risk = filtered_df.groupby('course')['risk_score'].mean().reset_index()
            course_avg_risk = course_avg_risk.sort_values('risk_score', ascending=False)
            for _, row in course_avg_risk.head(3).iterrows():
                if row['risk_score'] > 50:
                    st.write(f"• **{row['course']}**: High risk score ({row['risk_score']:.0f}). Review course content.")
        else:
            st.write("Not enough data for course recommendations.")

def show_faq():
    """Menampilkan halaman FAQ - Lengkap & Informatif"""
    
    st.markdown('<p class="main-header"><i class="fas fa-circle-question"></i> FAQ - Pusat Bantuan AILA</p>', unsafe_allow_html=True)
    st.markdown("Temukan jawaban atas pertanyaan yang paling sering ditanyakan seputar platform Quipper dan AILA.")
    st.markdown("---")
    
    search = st.text_input("🔍 Cari pertanyaan...", placeholder="Ketik kata kunci...")
    st.markdown("---")
    
    faqs = [
        # === JADWAL BELAJAR ===
        {
            "category": "📚 Jadwal Belajar",
            "question": "Bagaimana cara membuat jadwal belajar yang efektif?",
            "answer": "Kami merekomendasikan metode Pomodoro: belajar 25 menit, istirahat 5 menit. Lakukan 4-5 siklus per sesi. Prioritaskan materi yang sulit di pagi hari saat pikiran masih segar, dan gunakan jadwal yang konsisten setiap hari. Jangan lupa untuk memasukkan waktu istirahat dan refreshing."
        },
        {
            "category": "📚 Jadwal Belajar",
            "question": "Berapa lama waktu ideal belajar per hari untuk siswa SMA?",
            "answer": "Untuk siswa SMA, waktu belajar ideal adalah 3-4 jam per hari di luar jam sekolah. Bisa dibagi menjadi 2 sesi: pagi (1.5 jam) dan sore/malam (1.5-2 jam). Jangan lupa istirahat setiap 45-60 menit. Yang terpenting adalah kualitas, bukan kuantitas."
        },
        {
            "category": "📚 Jadwal Belajar",
            "question": "Bagaimana cara belajar yang efektif untuk menghadapi ujian?",
            "answer": "1. Buat ringkasan materi dari setiap bab, 2. Kerjakan soal-soal latihan dari tahun sebelumnya, 3. Gunakan teknik spaced repetition (ulangi materi secara berkala), 4. Diskusikan dengan teman untuk pemahaman yang lebih dalam, 5. Istirahat yang cukup sebelum ujian, 6. Jangan lupa berdoa dan tetap percaya diri."
        },
        {
            "category": "📚 Jadwal Belajar",
            "question": "Apa itu metode Pomodoro dan bagaimana cara menerapkannya?",
            "answer": "Metode Pomodoro adalah teknik manajemen waktu yang membagi waktu belajar menjadi interval 25 menit (disebut pomodoro) diikuti dengan istirahat 5 menit. Setelah 4 pomodoro, ambil istirahat panjang 15-30 menit. Cara menerapkannya: 1) Tentukan tugas, 2) Set timer 25 menit, 3) Fokus penuh, 4) Istirahat 5 menit, 5) Ulangi siklus."
        },
        {
            "category": "📚 Jadwal Belajar",
            "question": "Kapan waktu terbaik untuk belajar? Pagi atau malam?",
            "answer": "Setiap orang memiliki ritme yang berbeda. Namun secara umum, pagi hari (06.00-09.00) adalah waktu terbaik untuk belajar materi baru karena pikiran masih segar. Sore/malam (19.00-21.00) cocok untuk mengulang dan latihan soal. Temukan waktu yang paling produktif untuk Anda dan konsistenlah."
        },
        # === MODUL & MATERI ===
        {
            "category": "📖 Modul & Materi",
            "question": "Di mana saya bisa mengakses semua modul pembelajaran?",
            "answer": "Semua modul pembelajaran dapat diakses melalui dashboard Quipper School di menu Materi Belajar. Modul tersedia dalam bentuk video pembelajaran, PDF rangkuman, dan latihan interaktif. Pastikan Anda login dengan akun Quipper Anda."
        },
        {
            "category": "📖 Modul & Materi",
            "question": "Apakah Quipper menyediakan modul untuk persiapan UTBK?",
            "answer": "Ya! Quipper menyediakan paket khusus Quipper UTBK yang berisi ribuan soal prediksi terbaru, tryout online dengan sistem penilaian, video pembahasan dari tutor berpengalaman, dan materi khusus untuk setiap mata pelajaran UTBK. Akses melalui menu UTBK Center di dashboard utama."
        },
        {
            "category": "📖 Modul & Materi",
            "question": "Bagaimana cara mendownload materi untuk belajar offline?",
            "answer": "Untuk mendownload materi: 1) Buka modul yang diinginkan, 2) Klik tombol Download (ikon panah ke bawah), 3) File PDF atau video akan tersimpan di perangkat Anda. Materi yang sudah diunduh bisa diakses tanpa koneksi internet."
        },
        {
            "category": "📖 Modul & Materi",
            "question": "Apakah materi di Quipper selalu update dengan kurikulum terbaru?",
            "answer": "Ya, tim kurikulum Quipper secara rutin memperbarui materi sesuai dengan kurikulum terbaru (Kurikulum Merdeka). Setiap ada perubahan kebijakan dari Kemendikbud, materi akan disesuaikan dalam waktu maksimal 2 minggu."
        },
        {
            "category": "📖 Modul & Materi",
            "question": "Bagaimana cara mengetahui materi yang sudah saya pelajari?",
            "answer": "Anda dapat melihat progress belajar di dashboard utama. Setiap modul yang sudah Anda selesaikan akan ditandai dengan centang hijau. Di menu Progress Anda bisa melihat detail materi yang sudah dan belum dipelajari."
        },
        {
            "category": "📖 Modul & Materi",
            "question": "Apa yang harus dilakukan jika video pembelajaran tidak bisa diputar?",
            "answer": "1. Periksa koneksi internet, 2. Refresh halaman browser, 3. Coba gunakan browser lain (Chrome/Firefox), 4. Clear cache browser, 5. Coba akses di jam berbeda. Jika masih bermasalah, laporkan melalui menu Bantuan."
        },
        # === REKOMENDASI ===
        {
            "category": "🎯 Rekomendasi Belajar",
            "question": "Bagaimana cara mendapatkan rekomendasi materi yang sesuai?",
            "answer": "AILA (AI Learning Analytics) akan menganalisis performa Anda dan memberikan rekomendasi materi yang sesuai. Cek bagian Rekomendasi Belajar di dashboard utama. Sistem akan merekomendasikan materi yang perlu ditingkatkan dan modul selanjutnya yang cocok."
        },
        {
            "category": "🎯 Rekomendasi Belajar",
            "question": "Materi apa yang sebaiknya saya pelajari pertama kali?",
            "answer": "Mulailah dengan materi yang menjadi dasar (fundamental) untuk mata pelajaran yang Anda ambil. Contoh: untuk Matematika, kuasai aljabar dasar sebelum kalkulus. Dashboard AILA akan menunjukkan area yang perlu Anda tingkatkan."
        },
        {
            "category": "🎯 Rekomendasi Belajar",
            "question": "Bagaimana cara meningkatkan nilai quiz saya secara signifikan?",
            "answer": "1. Review materi sebelum quiz, 2. Kerjakan latihan soal tambahan, 3. Perhatikan feedback dari quiz sebelumnya, 4. Tonton ulang video pembahasan, 5. Catat kesalahan dan pelajari kembali, 6. Konsistensi adalah kunci!"
        },
        {
            "category": "🎯 Rekomendasi Belajar",
            "question": "Bagaimana cara mengatasi rasa malas belajar?",
            "answer": "1. Mulai dari hal kecil (belajar 10 menit dulu), 2. Buat target realistis, 3. Beri reward setelah mencapai target, 4. Cari teman belajar, 5. Ubah mindset bahwa belajar adalah investasi masa depan, 6. Gunakan teknik Pomodoro."
        },
        # === BANTUAN & DUKUNGAN ===
        {
            "category": "💬 Bantuan & Dukungan",
            "question": "Bagaimana cara menghubungi mentor jika saya kesulitan?",
            "answer": "Anda dapat menghubungi mentor melalui: 1) Fitur chat di dashboard Quipper, 2) Menu Bantuan, 3) Email ke support@quipper.com. Mentor akan merespon dalam waktu 1x24 jam."
        },
        {
            "category": "💬 Bantuan & Dukungan",
            "question": "Apa yang harus dilakukan jika modul tidak bisa diakses?",
            "answer": "1. Periksa koneksi internet, 2. Refresh halaman (F5), 3. Coba browser lain, 4. Clear cache browser, 5. Coba di perangkat lain, 6. Hubungi tim teknis Quipper jika masih bermasalah."
        },
        {
            "category": "💬 Bantuan & Dukungan",
            "question": "Bagaimana cara melaporkan bug atau error?",
            "answer": "Anda dapat melaporkan bug melalui: 1) Menu Bantuan -> Laporkan Masalah, 2) Email ke support@quipper.com, 3) Chat dengan tim support. Sertakan screenshot dan deskripsi detail masalah."
        },
        {
            "category": "💬 Bantuan & Dukungan",
            "question": "Apakah Quipper menyediakan layanan konsultasi privat?",
            "answer": "Ya, Quipper menyediakan layanan konsultasi privat dengan mentor berpengalaman. Anda bisa memilih jadwal dan topik yang ingin didiskusikan melalui menu Konsultasi di dashboard."
        },
        {
            "category": "💬 Bantuan & Dukungan",
            "question": "Bagaimana cara bergabung dengan komunitas belajar Quipper?",
            "answer": "Anda bisa bergabung dengan komunitas belajar Quipper melalui: 1) Grup diskusi di forum Quipper, 2) Komunitas Telegram resmi, 3) Event webinar dan workshop. Informasi lebih lanjut di menu Komunitas."
        },
        # === AILA DASHBOARD ===
        {
            "category": "🤖 AILA Dashboard",
            "question": "Apa itu AILA dan apa fungsinya?",
            "answer": "AILA (AI Learning Analytics) adalah dashboard berbasis AI yang membantu tim Quipper memantau performa siswa secara real-time. Fungsinya: mendeteksi siswa berisiko dropout, menganalisis sentimen feedback, dan memberikan rekomendasi intervensi personal."
        },
        {
            "category": "🤖 AILA Dashboard",
            "question": "Siapa yang bisa mengakses AILA Dashboard?",
            "answer": "AILA Dashboard dirancang untuk internal tim Quipper, terutama Academic Operations Team, Management, dan Mentor. Dashboard ini tidak diakses langsung oleh siswa (student-facing)."
        },
        {
            "category": "🤖 AILA Dashboard",
            "question": "Bagaimana AILA mendeteksi siswa berisiko dropout?",
            "answer": "AILA menggunakan model machine learning (XGBoost) yang menganalisis berbagai faktor seperti frekuensi login, durasi belajar, progress modul, nilai quiz, dan aktivitas forum. Model ini memberikan risk score 0-100% untuk setiap siswa."
        }
    ]
    
    # Filter FAQ berdasarkan search
    if search:
        search_lower = search.lower()
        filtered_faqs = [f for f in faqs if search_lower in f['question'].lower() or search_lower in f['answer'].lower() or search_lower in f['category'].lower()]
    else:
        filtered_faqs = faqs
    
    st.markdown(f"**{len(filtered_faqs)}** pertanyaan ditemukan")
    
    if filtered_faqs:
        categories = {}
        for faq in filtered_faqs:
            cat = faq['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(faq)
        
        for category, items in categories.items():
            st.markdown(f"### {category}")
            for faq in items:
                with st.expander(f"❓ {faq['question']}"):
                    st.markdown(f"<div class='faq-answer'>{faq['answer']}</div>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("Tidak ada pertanyaan yang sesuai dengan pencarian Anda. Coba kata kunci lain.")

def main():
    df = load_data()
    
    dropout_predictor = DropoutPredictor()
    sentiment_analyzer = SentimentAnalyzer()
    
    try:
        dropout_predictor.predict(df)
    except:
        with st.spinner("⏳ Melatih model prediksi dropout..."):
            dropout_predictor.train(df)
        st.success("✅ Model siap digunakan!")
    
    # Sidebar filters
    st.sidebar.title("🎯 Filter Data")
    st.sidebar.markdown("---")
    
    courses = ['All'] + sorted(df['course'].unique().tolist())
    selected_course = st.sidebar.selectbox("📚 Select Course", courses)
    
    risk_levels = ['All', 'High', 'Medium', 'Low']
    selected_risk = st.sidebar.selectbox("⚠️ Risk Level", risk_levels)
    
    cities = ['All'] + sorted(df['city'].unique().tolist())
    selected_city = st.sidebar.selectbox("📍 City", cities)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 **Powered by AI**")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_course != 'All':
        filtered_df = filtered_df[filtered_df['course'] == selected_course]
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['risk_category'] == selected_risk]
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city'] == selected_city]
    
    if len(filtered_df) == 0:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih. Menampilkan semua data.")
        filtered_df = df.copy()
    
    total_students = len(filtered_df)
    high_risk = len(filtered_df[filtered_df['risk_category'] == 'High'])
    avg_progress = filtered_df['progress_rate'].mean() if total_students > 0 else 0
    avg_quiz = filtered_df['avg_quiz_score'].mean() if total_students > 0 else 0
    
    # TABS: Dashboard | FAQ
    tab1, tab2 = st.tabs(["📊 Dashboard", "❓ FAQ & Panduan"])
    
    with tab1:
        show_dashboard(filtered_df, dropout_predictor, sentiment_analyzer, total_students, high_risk, avg_progress, avg_quiz)
    
    with tab2:
        show_faq()

if __name__ == "__main__":
    main()
