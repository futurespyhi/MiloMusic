# Installation and Running

```bash
# 1. Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt

# 3.Using YuEGP(GPU Poor)
git clone https://github.com/deepbeepmeep/YuEGP.git
cd YuEGP
pip install -r requirements.txt

# Comment out vocoder && post_process_audio in gradio_server.py

# 4. Set up environment variables in .env file:
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key

# 5. Run the app(make sure you are under root)
python app.py
```

The app will launch in your web browser after running.
