# MiloMusic 🎵

[//]: # (<p align="center">)

[//]: # (  <img src="assets/milomusic-logo.png" alt="MiloMusic Logo" width="200"/>)

[//]: # (</p>)

## 🦙 AI-Powered Music Creation for Everyone

MiloMusic is an innovative platform that leverages multiple AI models to democratize music creation. Whether you're a seasoned musician or have zero musical training, MiloMusic enables you to create high-quality, lyrics-focused music through natural language conversation.

> A platform for everyone - regardless of musical training at the intersection of AI and creative expression.

## 🚀 Features

- **Natural Language Interface** - Just start talking to generate song lyrics
- **Genre & Mood Selection** - Customize your music with different genres and moods
- **Iterative Creation Process** - Refine your lyrics through conversation
- **High-Quality Music Generation** - Transform lyrics into professional-sounding music
- **User-Friendly Interface** - Intuitive UI built with Gradio

## 🔧 Architecture

MiloMusic employs a sophisticated multi-model pipeline to deliver a seamless music creation experience:

### Phase 1: Lyrics Generation
1. **Speech-to-Text** - User voice input is transcribed using `gpt4o-transcribe`
2. **Conversation & Refinement** - `llama-4-scout-17b-16e-instruct` handles the creative conversation, generates lyrics based on user requests, and allows for iterative refinement

### Phase 2: Music Generation
1. **Lyrics Structuring** - `Gemini flash 2.0` processes the conversation history and structures the final lyrics for music generation
2. **Music Synthesis** - `YuE` (乐) transforms the structured lyrics into complete songs with vocals and instrumentation

## 💻 Technical Stack

- **LLM Models**:
  - `gpt4o-transcribe` - For speech-to-text conversion
  - `llama-4-scout-17b-16e-instruct` - For creative conversation and lyrics generation
  - `Gemini flash 2.0` - For lyrics structuring
  - `YuE` - For music generation
- **UI**: Gradio
- **Deployment**: Support for H100 GPU with 80GB VRAM (full model) or RTX 4090 with 24GB VRAM (quantized model)

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (24GB+ VRAM recommended)
- Dependencies listed in `requirements.txt`

## 🔍 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/futurespyhi/MiloMusic.git
cd MiloMusic

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Using the Interface:
1. Select your genre, mood, and theme preferences
2. Start talking about your song ideas
3. The assistant will create lyrics based on your selections
4. Give feedback to refine the lyrics
5. When you're happy with the lyrics, click "Generate Music from Lyrics"
6. Listen to your generated song!

## 🔬 Performance

MiloMusic can run in two configurations:

- **Full Model** (H100 GPU with 80GB VRAM): Generates music in ~9 minutes
- **Quantized Model** (RTX 4090 with 24GB VRAM): Generates music in ~13 minutes

## 🛠️ Development Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Multi-model & multi-API setup | Implemented seamless integration between different AI models and APIs |
| Compatibility issues | Developed custom adapters for model interoperability |
| Gradio limitations | Extended Gradio code to overcome structural design issues |
| Computational constraints | Implemented model quantization to support consumer GPUs |

## 🔮 Future Improvements

- Implement a wait queue system to improve UX during music generation
- Explore faster model alternatives for reduced generation time
- Add more customization options for music style and composition
- Implement batch processing for multiple song generation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Team

- Norton Gu
- Anakin Huang
- Erik Wasmosy

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  Made with ❤️ and 🦙 (LLaMA)
</p>
