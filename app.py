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
from models.chatbot import StudentAssistantChatbot

# Page configuration
st.set_page_config(
    page_title="Quipper AI Learning Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - MERAH TELKOM
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2.5rem;
        color: #E31E24;
        margin-bottom: 0;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    /* Sidebar Styling - GELAP dengan teks PUTIH */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding-top: 20px;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stSelectbox div {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background-color: rgba(255,255,255,0.12) !important;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]:hover {
        background-color: rgba(255,255,255,0.2) !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
        color: white !important;
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
    
    /* Metric Cards - Elegan */
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
    
    /* Risk Colors - JELAS */
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
    
    /* Chat Messages */
    .chat-message-user {
        background-color: #f0f0f0;
        padding: 10px 14px;
        border-radius: 12px;
        margin: 5px 0;
        border-left: 4px solid #E31E24;
    }
    .chat-message-bot {
        background-color: #fff5f5;
        padding: 10px 14px;
        border-radius: 12px;
        margin: 5px 0;
        border-left: 4px solid #28a745;
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
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #E31E24;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #B71C1C;
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
    """Generate personalized intervention message"""
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

def main():
    # Load data
    df = load_data()
    
    # Initialize models
    dropout_predictor = DropoutPredictor()
    sentiment_analyzer = SentimentAnalyzer()
    chatbot = StudentAssistantChatbot()
    
    # Train dropout model if not exists
    try:
        dropout_predictor.predict(df)
    except:
        with st.spinner("⏳ Melatih model prediksi dropout... (hanya sekali)"):
            dropout_predictor.train(df)
        st.success("✅ Model siap digunakan!")
    
    # Sidebar filters - GELAP dengan teks PUTIH
    st.sidebar.title("🎯 Filter Data")
    st.sidebar.markdown("---")
    
    # Course filter
    courses = ['All'] + sorted(df['course'].unique().tolist())
    selected_course = st.sidebar.selectbox("📚 Select Course", courses)
    
    # Risk level filter
    risk_levels = ['All', 'High', 'Medium', 'Low']
    selected_risk = st.sidebar.selectbox("⚠️ Risk Level", risk_levels)
    
    # City filter
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
    
    # Main content
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<p class="main-header">🎓 AI Learning Analytics Dashboard</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p class="sub-header" style="text-align: right;">Quipper | {datetime.now().strftime("%d %B %Y")}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # KPI Metrics
    st.subheader("📊 Key Performance Indicators")
    
    total_students = len(filtered_df)
    high_risk = len(filtered_df[filtered_df['risk_category'] == 'High'])
    avg_progress = filtered_df['progress_rate'].mean()
    avg_quiz = filtered_df['avg_quiz_score'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total_students}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ High Risk Students</div>
            <div class="metric-value risk-high">{high_risk}</div>
            <div class="metric-sub">{high_risk/total_students*100:.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Average Progress</div>
            <div class="metric-value">{avg_progress:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📝 Average Quiz Score</div>
            <div class="metric-value">{avg_quiz:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Analysis
    st.subheader("🎯 Student Risk Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        risk_counts = filtered_df['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        colors = {'High': '#E31E24', 'Medium': '#FF6B00', 'Low': '#28a745'}
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
                     color='Avg Risk Score',
                     color_continuous_scale='Reds')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # High Risk Students
    st.subheader("📋 High Risk Students - Priority Intervention")
    
    high_risk_students = filtered_df[filtered_df['risk_category'] == 'High'].copy()
    high_risk_students = high_risk_students.sort_values('risk_score', ascending=False)
    
    if len(high_risk_students) > 0:
        risk_factors = []
        for _, student in high_risk_students.head(10).iterrows():
            factors = dropout_predictor.get_risk_factors(student)
            risk_factors.append(', '.join(factors) if factors else 'Multiple risk indicators')
        
        display_df = high_risk_students.head(10)[['student_id', 'name', 'course', 'risk_score', 'progress_rate', 'avg_quiz_score', 'days_active']].copy()
        display_df['risk_factors'] = risk_factors[:len(display_df)]
        
        # TAMPILKAN TABEL DENGAN WARMA - hanya risk_score yang >70 berwarna merah
        st.dataframe(
            display_df.style.applymap(
                lambda x: 'color: #E31E24; font-weight: bold;' if isinstance(x, (int, float)) and x > 70 else '',
                subset=['risk_score']
            ),
            use_container_width=True
        )
        
        st.info("💡 **AI Intervention Suggestions:**")
        for _, student in high_risk_students.head(3).iterrows():
            factors = dropout_predictor.get_risk_factors(student)
            message = generate_intervention_message(student, factors)
            st.write(f"• **{student['name']}** (ID: {student['student_id']}): {message}")
    else:
        st.success("✅ No high risk students found! Great job!")
    
    st.markdown("---")
    
    # Sentiment Analysis
    st.subheader("💬 Student Sentiment Analysis")
    
    if 'feedback_text' in filtered_df.columns and len(filtered_df) > 0:
        results = sentiment_analyzer.analyze_batch(filtered_df['feedback_text'].tolist())
        filtered_df['predicted_sentiment'] = [r['sentiment'] for r in results]
        filtered_df['polarity_score'] = [r['polarity'] for r in results]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            sent_counts = filtered_df['predicted_sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Count']
            colors2 = {'positive': '#28a745', 'negative': '#E31E24', 'neutral': '#FF6B00'}
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
    
    # FEEDBACK SUMMARY
    st.subheader("📝 Student Feedback Summary Report")
    
    if 'feedback_text' in filtered_df.columns and len(filtered_df) > 0:
        # Get sentiment summary
        summary = sentiment_analyzer.get_sentiment_summary(filtered_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9rem; color:#666;">Total Feedback</div>
                <div class="metric-value">{summary['total_analyzed']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_feedback = summary['total_analyzed']
            positive_pct = (summary['positive'] / total_feedback * 100) if total_feedback > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9rem; color:#666;">Positive Feedback</div>
                <div class="metric-value" style="color:#28a745;">{summary['positive']}</div>
                <div style="font-size:0.8rem; color:#666;">{positive_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_feedback = summary['total_analyzed']
            negative_pct = (summary['negative'] / total_feedback * 100) if total_feedback > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.9rem; color:#666;">Negative Feedback</div>
                <div class="metric-value" style="color:#E31E24;">{summary['negative']}</div>
                <div style="font-size:0.8rem; color:#666;">{negative_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Feedback details table
        st.subheader("📋 Detailed Feedback Analysis")
        
        # Create feedback summary table
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
            
            # Color coding for sentiment
            def color_sentiment(val):
                if val == 'POSITIVE':
                    return 'color: #28a745'
                elif val == 'NEGATIVE':
                    return 'color: #E31E24'
                else:
                    return 'color: #FF6B00'
            
            st.dataframe(
                feedback_df.style.map(color_sentiment, subset=['Sentiment']),
                use_container_width=True,
                height=300
            )
            
            # Export button
            csv = feedback_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feedback Summary (CSV)",
                data=csv,
                file_name=f"feedback_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Top Keywords Summary
        if 'top_keywords' in summary and summary['top_keywords']:
            st.subheader("🔑 Top Keywords from Feedback")
            cols = st.columns(5)
            for idx, (word, count) in enumerate(summary['top_keywords'][:5]):
                if idx < 5:
                    cols[idx].metric(label=word.capitalize(), value=count)
    else:
        st.info("No feedback data available for analysis")
    
    st.markdown("---")
    
    # Learning Progress
    st.subheader("📈 Learning Progress Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig5 = px.histogram(filtered_df, x='progress_rate', 
                           title='Student Progress Distribution',
                           nbins=20,
                           color_discrete_sequence=['#E31E24'])
        fig5.update_layout(height=350)
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        fig6 = px.scatter(filtered_df, x='progress_rate', y='avg_quiz_score',
                         color='risk_category',
                         title='Quiz Score vs Progress',
                         color_discrete_map={'High': '#E31E24', 'Medium': '#FF6B00', 'Low': '#28a745'},
                         hover_data=['student_id', 'name'],
                         size='total_logins')
        fig6.update_layout(height=350)
        st.plotly_chart(fig6, use_container_width=True)
    
    # AI Recommendations
    st.subheader("🤖 AI Learning Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Dashboard AI Insights:**")
        insights = []
        
        if high_risk > total_students * 0.2:
            insights.append(f"⚠️ {high_risk/total_students*100:.1f}% of students are at high risk. Immediate intervention recommended.")
        
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
        st.markdown("**📚 Course Improvement Recommendations:**")
        if len(filtered_df) > 10:
            course_avg_risk = filtered_df.groupby('course')['risk_score'].mean().reset_index()
            course_avg_risk = course_avg_risk.sort_values('risk_score', ascending=False)
            for _, row in course_avg_risk.head(3).iterrows():
                if row['risk_score'] > 50:
                    st.write(f"• **{row['course']}**: High risk score ({row['risk_score']:.0f}). Review course content.")
    
    st.markdown("---")
    
    # ============= CHATBOT ASSISTANT =============
    st.subheader("🤖 AI Chatbot Assistant")
    
    # Create two columns: chat history and input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat history display
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.chat_history.append({
                "role": "assistant",
                "message": "Halo! Saya asisten belajar AI. Ada yang bisa saya bantu? 😊"
            })
        
        # Display chat messages
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message-user">
                    <b>🧑‍🎓 Kamu:</b> {msg['message']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message-bot">
                    <b>🤖 Bot:</b> {msg['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = [{
                "role": "assistant",
                "message": "Halo! Saya asisten belajar AI. Ada yang bisa saya bantu? 😊"
            }]
    
    with col2:
        st.markdown("**💡 Quick Questions:**")
        quick_questions = [
            "Apa rekomendasi belajarku?",
            "Bagaimana progress saya?",
            "Semangat dong!",
            "Materi Matematika",
            "Terimakasih!"
        ]
        
        for q in quick_questions:
            if st.button(q, key=f"q_{q}"):
                # Get student data for context
                student_context = None
                if not filtered_df.empty:
                    high_risk = filtered_df[filtered_df['risk_category'] == 'High']
                    if not high_risk.empty:
                        student_context = high_risk.iloc[0].to_dict()
                    else:
                        student_context = filtered_df.iloc[0].to_dict()
                
                # Get response
                response = chatbot.get_response(q, student_context)
                
                # Add to history
                st.session_state.chat_history.append({"role": "user", "message": q})
                st.session_state.chat_history.append({"role": "assistant", "message": response})
    
    # Chat input
    user_input = st.text_input("Tanyakan sesuatu ke asisten belajar:", key="chat_input")
    if st.button("Kirim") and user_input:
        # Get student data for context
        student_context = None
        if not filtered_df.empty:
            high_risk = filtered_df[filtered_df['risk_category'] == 'High']
            if not high_risk.empty:
                student_context = high_risk.iloc[0].to_dict()
            else:
                student_context = filtered_df.iloc[0].to_dict()
        
        # Get response
        response = chatbot.get_response(user_input, student_context)
        
        # Add to history
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "assistant", "message": response})
    
    st.markdown("---")
    st.markdown("*💡 Powered by AI Learning Analytics. Data updated in real-time.*")

if __name__ == "__main__":
    main()
