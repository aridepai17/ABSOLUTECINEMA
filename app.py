import gradio as gr
import numpy as np
from deepface import DeepFace
import requests
import os
import random
import cv2
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from gradio.themes.utils import fonts
from gradio.themes.base import Base

class EmotionMovieBot:
    def __init__(self):
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")
        if not self.tmdb_api_key:
            raise ValueError("TMDB_API_KEY environment variable is required")
        
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        
        # Initialize LangChain components
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY not found. LangChain features will be disabled.")
            self.llm_enabled = False
        else:
            self.llm_enabled = True
            self.llm = OpenAI(
                temperature=0.7,
                max_tokens=200,
                openai_api_key=openai_api_key
            )
            
            # Enhanced description prompt
            self.description_prompt = PromptTemplate(
                input_variables=["title", "genre_names", "original_plot", "emotion", "rating"],
                template="""
You are a movie enthusiast writing for someone who is currently feeling {emotion}.
Movie: {title}
Genres: {genre_names}
Rating: {rating}/10
Original Description: {original_plot}

Rewrite this movie description in 2-3 sentences to appeal specifically to someone feeling {emotion}. 
Make it emotionally engaging and explain why this movie would be perfect for their current mood.
Focus on the emotional journey and experience they'll have watching it.

Enhanced Description:"""
            )
            
            # Movie reasoning prompt
            self.reasoning_prompt = PromptTemplate(
                input_variables=["title", "emotion", "confidence"],
                template="""
Explain in 1-2 sentences why "{title}" is specifically recommended for someone feeling {emotion} 
with {confidence}% confidence. Focus on the therapeutic or mood-enhancing benefits.

Reasoning:"""
            )
            
            self.description_chain = LLMChain(llm=self.llm, prompt=self.description_prompt)
            self.reasoning_chain = LLMChain(llm=self.llm, prompt=self.reasoning_prompt)
        
        # Emotion-to-genre mapping based on psychological research
        self.emotion_genre_map = {
            "happy": [35, 10749, 16],  # Comedy, Romance, Animation
            "sad": [18, 10402],        # Drama, Music
            "angry": [28, 53],         # Action, Thriller
            "fear": [27, 9648],        # Horror, Mystery
            "surprise": [878, 14],     # Sci-Fi, Fantasy
            "disgust": [80, 9648],     # Crime, Mystery
            "neutral": [35, 28, 18]    # Comedy, Action, Drama
        }
        
        # Genre ID to name mapping for LangChain context
        self.genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
            99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
            27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        
        self.emotion_context = {
            "happy": "You seem to be in a great mood! Perfect time for some uplifting entertainment.",
            "sad": "I notice you might be feeling a bit down. Here are some movies that might help you process emotions or lift your spirits.",
            "angry": "Feeling intense? These high-energy films might be just what you need.",
            "fear": "In the mood for some thrills? These carefully selected films will give you the right amount of suspense.",
            "surprise": "You look intrigued! These mind-bending films will keep you guessing.",
            "disgust": "Looking for something different? These compelling stories might change your perspective.",
            "neutral": "Ready for some great cinema? Here are some universally acclaimed films."
        }
    
    def detect_emotion_from_webcam(self, image):
        """Detect emotion from webcam image using DeepFace with OpenCV preprocessing"""
        try:
            # Convert PIL image to numpy array
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
                img_array = np.array(image)
            else:
                img_array = np.array(image)
            
            # Use OpenCV for face detection preprocessing
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # No face detected, try with the original image
                result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
            else:
                # Face detected, analyze emotion
                result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=True)
            
            # Handle both list and dict responses
            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
            
            return dominant_emotion, emotions[dominant_emotion], emotions
            
        except Exception as e:
            print(f"Emotion detection error: {str(e)}")
            # Return neutral emotion as fallback
            return "neutral", 0.5, {"neutral": 0.5, "happy": 0.1, "sad": 0.1, "angry": 0.1, "fear": 0.1, "surprise": 0.1, "disgust": 0.1}
    
    def get_movie_recommendations_from_tmdb(self, emotion):
        """Get movie recommendations based on emotion using TMDB API"""
        try:
            genre_ids = self.emotion_genre_map.get(emotion.lower(), [35, 28, 18])
            all_movies = []

            pages_to_fetch = 5 # 5 pages x 20 results = up to 100 movies


            for page in range(1, pages_to_fetch + 1):
                params = {
                    "api_key": self.tmdb_api_key,
                    "with_genres": ",".join(map(str, genre_ids)),
                    "sort_by": "popularity.desc",
                    "vote_average.gte": 6.5,
                    "vote_count.gte": 100,
                    "adult": False,
                    "page": page
                }
            
                response = requests.get(f"{self.tmdb_base_url}/discover/movie", params=params, timeout=10)
                response.raise_for_status()
            
                movies = response.json().get('results', [])
                all_movies.extend(movies)

            if len(all_movies) >= 5:
                return random.sample(all_movies, 5)
            else:
                return all_movies # return fewer if less than 5 found
            
        except requests.exceptions.RequestException as e:
            print(f"TMDB API error: {str(e)}")
            return self._get_fallback_movies(emotion)
        except Exception as e:
            print(f"Unexpected error fetching movies: {str(e)}")
            return self._get_fallback_movies(emotion)
    
    def _get_fallback_movies(self, emotion):
        """Fallback movie recommendations when API fails"""
        fallback_movies = {
            "happy": [
                {"title": "The Pursuit of Happyness", "overview": "A heartwarming story of determination and hope.", "vote_average": 8.0, "release_date": "2006-12-15"},
                {"title": "Finding Nemo", "overview": "An adventurous journey of a father fish searching for his son.", "vote_average": 8.2, "release_date": "2003-05-30"}
            ],
            "sad": [
                {"title": "Inside Out", "overview": "A beautiful exploration of emotions and growing up.", "vote_average": 8.1, "release_date": "2015-06-19"},
                {"title": "The Shawshank Redemption", "overview": "A story of hope and friendship in the darkest places.", "vote_average": 9.3, "release_date": "1994-09-23"}
            ],
            "angry": [
                {"title": "Mad Max: Fury Road", "overview": "High-octane action in a post-apocalyptic wasteland.", "vote_average": 8.1, "release_date": "2015-05-15"},
                {"title": "John Wick", "overview": "Stylized action and revenge story.", "vote_average": 7.4, "release_date": "2014-10-24"}
            ],
            "neutral": [
                {"title": "Inception", "overview": "A mind-bending thriller about dreams within dreams.", "vote_average": 8.8, "release_date": "2010-07-16"},
                {"title": "The Dark Knight", "overview": "A superhero film that transcends the genre.", "vote_average": 9.0, "release_date": "2008-07-18"}
            ]
        }
        
        return fallback_movies.get(emotion.lower(), fallback_movies["neutral"])
    
    def enhance_movie_description_with_langchain(self, movie, emotion, confidence):
        """Use LangChain to create enhanced, emotion-specific movie descriptions"""
        if not self.llm_enabled:
            return self._create_basic_description(movie, emotion, confidence)
        
        try:
            title = movie.get('title', 'Unknown Title')
            original_plot = movie.get('overview', 'No description available.')
            rating = movie.get('vote_average', 0)
            genre_ids = movie.get('genre_ids', [])
            genre_names = ', '.join([self.genre_map.get(gid, 'Unknown') for gid in genre_ids])
            
            # Generate enhanced description
            enhanced_description = self.description_chain.run(
                title=title,
                genre_names=genre_names,
                original_plot=original_plot,
                emotion=emotion,
                rating=rating
            ).strip()
            
            # Generate reasoning
            reasoning = self.reasoning_chain.run(
                title=title,
                emotion=emotion,
                confidence=int(confidence * 100)
            ).strip()
            
            return enhanced_description, reasoning
            
        except Exception as e:
            print(f"LangChain enhancement failed: {str(e)}")
            return self._create_basic_description(movie, emotion, confidence)
    
    def _create_basic_description(self, movie, emotion, confidence):
        """Fallback description when LangChain is not available"""
        title = movie.get('title', 'Unknown Title')
        original_plot = movie.get('overview', 'No description available.')
        genre_ids = movie.get('genre_ids', [])
        rating = movie.get('vote_average', 0)
        
        # Get genre names for this specific movie
        movie_genres = [self.genre_map.get(gid, 'Unknown') for gid in genre_ids[:3]]
        genre_text = ', '.join(movie_genres) if movie_genres else 'Various genres'
        
        emotion_enhancers = {
            "happy": "This uplifting film will complement your positive mood perfectly.",
            "sad": "This emotionally rich story might provide comfort and catharsis.",
            "angry": "This intense film can help channel your energy productively.",
            "fear": "This thrilling experience will give you controlled excitement.",
            "surprise": "This unpredictable story will keep you engaged and intrigued.",
            "disgust": "This compelling narrative might offer a fresh perspective.",
            "neutral": "This well-crafted film offers excellent entertainment value."
        }
        
        enhanced_description = f"{original_plot} {emotion_enhancers.get(emotion, emotion_enhancers['neutral'])}"
        
        # Create unique reasoning for each movie
        reasoning_templates = [
            f"'{title}' is perfect for your {emotion} mood because its {genre_text.lower()} elements (rated {rating}/10) create the ideal emotional experience.",
            f"With a {rating}/10 rating, '{title}' offers {genre_text.lower()} storytelling that resonates perfectly with your {emotion} emotional state.",
            f"The {genre_text.lower()} nature of '{title}' (rated {rating}/10) makes it an excellent choice for someone feeling {emotion}.",
            f"'{title}' combines {genre_text.lower()} with emotional depth (rated {rating}/10), making it ideal for your current {emotion} mood.",
            f"Based on your {emotion} emotion, '{title}' offers the perfect {genre_text.lower()} experience with its {rating}/10 rating."
        ]
        
        # Use movie title hash to select consistent but different reasoning for each movie
        reasoning_index = hash(title) % len(reasoning_templates)
        reasoning = reasoning_templates[reasoning_index]
        
        return enhanced_description, reasoning
    
    def generate_personalized_recommendations(self, emotion, movies, emotion_confidence):
        """Generate personalized recommendation display"""
        context = self.emotion_context.get(emotion.lower(), self.emotion_context["neutral"])
        
        recommendation_text = f"""🎬 **Real-Time Emotion-Based Movie Recommendations: {emotion.title()}**
        
{context}
**Emotion Detection Confidence:** {emotion_confidence:.1%}
**Powered by:** DeepFace + OpenCV + TMDB API + LangChain
**My Top Picks for You:**
"""
        
        selected_movies = random.sample(movies, min(5, len(movies)))
        
        for i, movie in enumerate(selected_movies, 1):
            title = movie.get('title', 'Unknown Title')
            rating = movie.get('vote_average', 0)
            poster_path = movie.get('poster_path', '')
            release_date = movie.get('release_date', 'Unknown')
            
            # Get enhanced description and reasoning
            enhanced_description, reasoning = self.enhance_movie_description_with_langchain(movie, emotion, emotion_confidence)
            
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
            year = release_date.split('-')[0] if release_date and release_date != 'Unknown' else 'Unknown'
            
            recommendation_text += f"""
<details style="margin: 15px 0; border: 2px solid #404040; border-radius: 12px; padding: 0; background: linear-gradient(135deg, #2a2a2a, #1f1f1f); box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
<summary style="cursor: pointer; font-size: 18px; font-weight: bold; padding: 15px 20px; background: linear-gradient(135deg, #4a90e2, #357abd); color: white; border-radius: 10px; margin: 0;">
🎬 {i}. {title} ({year}) ⭐ {rating}/10
</summary>
<div style="padding: 20px;">
"""
            
            if poster_url:
                recommendation_text += f"""
<div style="text-align: center; margin: 20px 0;">
<img src="{poster_url}" alt="{title} Poster" style="width: 200px; height: 300px; object-fit: cover; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.5);">
</div>
"""
            
            recommendation_text += f"""
<div style="background: linear-gradient(135deg, #333333, #2a2a2a); padding: 20px; border-radius: 10px; border-left: 4px solid #4a90e2;">
<h4 style="color: #4a90e2; margin-top: 0;">🤖 {"AI-Enhanced" if self.llm_enabled else "Emotion-Matched"} Description:</h4>
<p style="line-height: 1.7; color: #e0e0e0; margin: 0 0 15px 0; font-style: italic; background: rgba(74, 144, 226, 0.1); padding: 10px; border-radius: 5px;">{enhanced_description}</p>
<h4 style="color: #4a90e2; margin-top: 15px;">💡 Perfect for Your Current Mood:</h4>
<p style="line-height: 1.6; color: #c0c0c0; margin: 0;">{reasoning}</p>
</div>
</div>
</details>
"""
        
        recommendation_text += f"""
---
<div style="background: linear-gradient(135deg, #2a2a2a, #1f1f1f); padding: 20px; border-radius: 10px; border: 1px solid #404040; margin-top: 20px;">
<p style="color: #e0e0e0; margin: 0 0 10px 0;">🤖 <strong style="color: #4a90e2;">Tech Stack:</strong> DeepFace facial emotion recognition, OpenCV for real-time processing, TMDB API for movie data{", LangChain + OpenAI for AI enhancement" if self.llm_enabled else ""}</p>
<p style="color: #e0e0e0; margin: 0;">🎭 <strong style="color: #4a90e2;">Emotion Analysis:</strong> Detected {emotion} with {emotion_confidence:.1%} confidence from your webcam input</p>
</div>
"""
        
        return recommendation_text


def analyze_emotion_and_recommend(image):
    """Main function to process webcam input and generate recommendations"""
    if image is None:
        return "Please capture an image from your webcam first!", "", "❌ No image provided"
    
    try:
        bot = EmotionMovieBot()
        
        # Detect emotion from webcam input
        emotion, raw_confidence, all_emotions = bot.detect_emotion_from_webcam(image)
        
        # Normalize emotion scores for better confidence calculation
        total_score = sum(all_emotions.values())
        normalized_emotions = {k: v / total_score if total_score > 0 else 0.0 for k, v in all_emotions.items()}
        confidence = normalized_emotions[emotion]
        
        # Get movie recommendations from TMDB
        movies = bot.get_movie_recommendations_from_tmdb(emotion)
        
        if not movies:
            return "Error fetching movie recommendations", "Unable to get recommendations from TMDB API", "❌ API Error"
        
        # Generate personalized recommendations
        recommendation = bot.generate_personalized_recommendations(emotion, movies, confidence)
        
        # Create emotion analysis display
        emotion_analysis = f"""**Real-Time Emotion Detection Results:**
**Primary Emotion: {emotion.title()}** (Confidence: {confidence * 100:.1f}%)

**Detailed Emotion Analysis:**
"""
        for emo, score in sorted(normalized_emotions.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 20)  # Visual bar representation
            bar = "█" * bar_length + "░" * (20 - bar_length)
            emotion_analysis += f"- {emo.title()}: {score * 100:.1f}% {bar}\n"
        
        status = f"✅ Successfully analyzed emotion and generated recommendations! Detected: {emotion.title()} ({confidence * 100:.1f}%)"
        return emotion_analysis, recommendation, status
        
    except ValueError as e:
        if "TMDB_API_KEY" in str(e):
            return "Configuration Error", "TMDB API key is not configured. Please set the TMDB_API_KEY environment variable.", "❌ Configuration Error"
        else:
            return "Error", str(e), f"❌ Error: {str(e)}"
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        print(f"Full error: {e}")
        return "Error in processing", "Unable to analyze emotion or generate recommendations", error_msg


# Create the Gradio interface
def create_gradio_interface():
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        font=fonts.GoogleFont("Press Start 2P")
    )
    """Create and configure the Gradio interface"""
    with gr.Blocks(title="🎬 Real-Time Emotion-Based Movie Recommender", theme=custom_theme) as app:
        gr.Markdown("""
        # 🎬 ABSOLUTE CINEMA
        ### *Blume AI - Team 5*
        
        This system uses **DeepFace** for facial emotion recognition, **OpenCV** for real-time webcam processing, 
        **TMDB API** for movie data, and **LangChain** for AI-enhanced descriptions.
        
        **How it works:**
        1. 📸 Capture your facial expression using webcam
        2. 🧠 AI analyzes your emotion using DeepFace
        3. 🎬 System maps emotions to movie genres
        4. 🤖 LangChain generates personalized descriptions
        5. ✨ Get movies perfectly matched to your current mood!
        
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Real-Time Emotion Capture")
                
                webcam_input = gr.Image(
                    type="pil",
                    label="Webcam Input for Emotion Detection",
                    height=300,
                    sources=["webcam"]  # Only webcam for real-time capture
                )
                
                analyze_button = gr.Button(
                    "🔍 Analyze Emotion & Get Movie Recommendations", 
                    variant="primary", 
                    size="lg"
                )
                
                status_output = gr.Textbox(
                    label="📊 System Status",
                    value="Ready for real-time emotion analysis! Capture your image using webcam.",
                    interactive=False
                )
                
                emotion_display = gr.Textbox(
                    label="🎭 Current Emotion Status",
                    value="No emotion detected yet - capture an image to start!",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎬 Your Personalized Movie Recommendations")
                
                emotion_analysis_output = gr.Markdown(
                    value="Capture an image to see detailed emotion analysis with confidence scores...",
                    label="Emotion Analysis Results"
                )
                
                movie_recommendations_output = gr.Markdown(
                    value="Your emotion-based movie recommendations will appear here after analysis...",
                    label="Personalized Movie Recommendations"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 🔧 System Information & Tech Stack
                
                **Emotion Detection:** DeepFace (Facial Expression Analysis) + OpenCV (Real-time Processing)
                
                **Movie Data Source:** The Movie Database (TMDB) API
                
                **AI Enhancement:** LangChain + OpenAI (Emotion-specific descriptions)
                
                **Supported Emotions:** Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
                
                **Privacy:** All processing is real-time and images are not stored
                
                **Note:** Ensure good lighting for best emotion detection results
                """)
        
        # Event handlers
        analyze_button.click(
            fn=analyze_emotion_and_recommend,
            inputs=[webcam_input],
            outputs=[emotion_analysis_output, movie_recommendations_output, status_output]
        )
        
        webcam_input.change(
            fn=lambda img: "Image captured - ready for emotion analysis!" if img else "No image captured yet",
            inputs=[webcam_input],
            outputs=[emotion_display]
        )
        
        app.load(
            fn=lambda: "System ready! Capture your image to start real-time emotion-based movie recommendations 🎬😊",
            outputs=[emotion_display]
        )
    
    return app


# Launch the application
if __name__ == "__main__":
    try:
        app = create_gradio_interface()
        app.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            share=True,
            show_error=True
        )
    except Exception as e:
        print(f"Failed to launch application: {e}")
        print("Please ensure all environment variables are set correctly:")
        print("- TMDB_API_KEY: Your TMDB API key")
        print("- OPENAI_API_KEY: Your OpenAI API key (optional, for AI enhancement)")