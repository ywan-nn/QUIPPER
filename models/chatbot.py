import random
import re
from datetime import datetime

class StudentAssistantChatbot:
    def __init__(self):
        self.context = {}
        self.greetings = [
            "Halo! Ada yang bisa saya bantu? 😊",
            "Selamat datang! Saya asisten belajar kamu. Ada yang ingin ditanyakan?",
            "Hai! Siap membantu kamu belajar. Mau tanya apa?",
            "Hello! Senang bisa membantu kamu. Ada yang bisa saya bantu hari ini?"
        ]
        
        self.farewells = [
            "Sama-sama! Semangat terus belajarnya! 💪",
            "Senang bisa membantu! Jangan ragu tanya lagi ya.",
            "Terus semangat! Raih prestasi terbaikmu! 🎯",
            "Good luck! Kamu pasti bisa! 🚀"
        ]
        
        self.motivations = [
            "Kamu pasti bisa! Terus semangat! 💪",
            "Percaya sama kemampuanmu! 🎯",
            "Setiap langkah kecil itu penting. Tetap konsisten! 🌟",
            "Semua orang pernah kesulitan. Yang penting tetap berusaha!",
            "Belajar itu proses. Nikmati setiap momennya! 📚"
        ]
        
        self.default_responses = [
            "Hmm, menarik! Bisa ceritakan lebih detail?",
            "Saya belum begitu paham. Bisa diulangi?",
            "Coba tanyakan ke mentor ya, mereka pasti bisa bantu lebih detail.",
            "Saya masih belajar. Coba tanya yang lain? 😅"
        ]
    
    def get_response(self, user_input, student_data=None):
        """Main method to get chatbot response"""
        user_input = user_input.lower().strip()
        
        # Check for greetings
        if any(word in user_input for word in ['halo', 'hai', 'hi', 'selamat', 'pagi', 'siang', 'malam']):
            return random.choice(self.greetings)
        
        # Check for farewells
        if any(word in user_input for word in ['dadah', 'bye', 'sampai', 'jumpa', 'terimakasih', 'makasih']):
            return random.choice(self.farewells)
        
        # Check for motivation requests
        if any(word in user_input for word in ['semangat', 'motivasi', 'capek', 'lelah', 'pusing']):
            return random.choice(self.motivations)
        
        # Check for help with specific subjects
        if any(word in user_input for word in ['matematika', 'fisika', 'kimia', 'biologi', 'inggris']):
            return self._handle_subject_question(user_input)
        
        # Check for progress questions
        if any(word in user_input for word in ['progress', 'perkembangan', 'nilai', 'score', 'quiz']):
            return self._handle_progress_question(student_data)
        
        # Check for course recommendations
        if any(word in user_input for word in ['rekomendasi', 'saran', 'belajar apa', 'next']):
            return self._handle_recommendation(student_data)
        
        # Default response
        return random.choice(self.default_responses)
    
    def _handle_subject_question(self, user_input):
        """Handle subject-specific questions"""
        subjects = {
            'matematika': "📐 Matematika: Coba fokus ke rumus dasar dulu ya! Latihan soal setiap hari sangat membantu.",
            'fisika': "⚡ Fisika: Pahami konsepnya dulu, baru rumus. Banyak contoh soal bisa membantu.",
            'kimia': "🧪 Kimia: Hafalkan tabel periodik dan pahami reaksi dasarnya. Jangan lupa praktikum virtual!",
            'biologi': "🧬 Biologi: Banyak istilah yang perlu dihafal. Bikin mind map biar lebih mudah!",
            'inggris': "🇬🇧 Bahasa Inggris: Practice makes perfect! Coba tonton film atau baca artikel bahasa Inggris."
        }
        
        for subject, response in subjects.items():
            if subject in user_input:
                return response
        
        return "Mata pelajaran apa yang ingin kamu tanyakan? Ada Matematika, Fisika, Kimia, Biologi, atau Bahasa Inggris."
    
    def _handle_progress_question(self, student_data):
        """Handle progress-related questions"""
        if student_data is None:
            return "Untuk cek progress, silahkan lihat dashboard utama ya! 📊"
        
        name = student_data.get('name', 'Kamu')
        progress = student_data.get('progress_rate', 0)
        quiz_score = student_data.get('avg_quiz_score', 0)
        
        if progress < 30:
            return f"Halo {name}! Progress kamu baru {progress:.0f}%. Yuk kita mulai dari yang paling dasar dulu! Jangan menyerah ya 💪"
        elif progress < 60:
            return f"Progress kamu {progress:.0f}%. Lumayan! Coba tingkatkan lagi dengan latihan soal. Nilai quiz kamu {quiz_score:.0f}% 📈"
        else:
            return f"Wah, progress kamu sudah {progress:.0f}%! Kamu hebat! Pertahankan ya, sebentar lagi selesai 🎉"
    
    def _handle_recommendation(self, student_data):
        """Handle learning recommendations"""
        if student_data is None:
            return "Untuk rekomendasi belajar, coba cek dashboard dan lihat saran dari AI ya! 🤖"
        
        progress = student_data.get('progress_rate', 0)
        quiz_score = student_data.get('avg_quiz_score', 0)
        
        if progress < 30:
            return "📚 Rekomendasi: Mulai dari modul dasar dulu. Tonton video pembelajaran dan baca rangkuman materi."
        elif progress < 60 and quiz_score < 60:
            return "📚 Rekomendasi: Perbanyak latihan soal! Coba kerjakan quiz di setiap akhir modul."
        elif progress < 60 and quiz_score >= 60:
            return "📚 Rekomendasi: Lanjutkan ke modul berikutnya! Kamu sudah siap."
        else:
            return "📚 Rekomendasi: Kamu hampir selesai! Fokus ke latihan soal dan review materi yang masih sulit."
    
    def generate_intervention_message(self, student_data):
        """Generate proactive intervention message for at-risk students"""
        name = student_data.get('name', 'Siswa')
        course = student_data.get('course', '')
        progress = student_data.get('progress_rate', 0)
        risk_factors = student_data.get('risk_factors', [])
        
        message = f"Halo {name}! 👋\n\n"
        
        if progress < 20:
            message += f"Saya lihat kamu masih di awal perjalanan belajar {course}. Yuk kita mulai dari yang paling dasar dulu! "
        elif len(risk_factors) > 0 and 'quiz' in str(risk_factors[0]).lower():
            message += f"Materi {course} memang menantang, tapi kamu pasti bisa! Coba tonton video pembahasan soal-soal sebelumnya. "
        elif len(risk_factors) > 0 and 'activity' in str(risk_factors[0]).lower():
            message += f"Kelas {course} minggu ini seru banget lho, yuk masuk lagi! Ada materi baru yang menarik. "
        else:
            message += f"Bagaimana perkembangan belajar {course} hari ini? Ada yang bisa saya bantu? "
        
        message += "\n\nSemangat terus ya! 💪\n- Asisten Belajarmu"
        
        return message