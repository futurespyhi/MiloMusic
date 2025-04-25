import gradio as gr
import soundfile as sf
from dataclasses import dataclass, field
from typing import Any
import xxhash
import os
import groq
import tempfile
import numpy as np

from tools.groq_client import client as groq_client
import spaces

from tools.generate_lyrics import generate_structured_lyrics, format_lyrics_for_yue


@dataclass
class AppState:
    """
    Maintains the application state throughout user interactions.

    Stores conversation history, generation parameters, and processing flags
    to maintain context between different API calls and UI interactions.

    Attributes:
        conversation: List of message dictionaries in the chat history
        stopped: Flag indicating if audio recording has been stopped
        model_outs: Storage for any model outputs that need persistence
        lyrics: Current generated lyrics text
        genre: Selected musical genre for generation
        mood: Selected emotional mood for generation
        theme: Selected subject matter/theme for generation
    """
    conversation: list = field(default_factory=list)
    stopped: bool = False
    model_outs: Any = None
    lyrics: str = ""
    genre: str = "pop"
    mood: str = "upbeat"
    theme: str = "love"


def validate_api_keys():
    """
    Validate that all required API keys are properly loaded from environment variables.

    Checks for the presence of necessary API keys and logs their status.
    This helps prevent runtime errors when trying to access external services.

    Returns:
        bool: True if all required API keys are present, False otherwise
    """
    # Check GROQ API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY not found in environment variables!")
    else:
        print(f"GROQ API key successfully loaded: {groq_api_key[:5]}...")

    # Check GEMINI API key (if you use it)
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("WARNING: GEMINI_API_KEY not found in environment variables!")
    else:
        print(f"GEMINI API key successfully loaded: {gemini_api_key[:5]}...")

    # Add validation for other API keys if needed

    # Return validation result
    return groq_api_key is not None and gemini_api_key is not None


def process_whisper_response(completion):
    """
    Process Whisper transcription response and filter out silence.

    Analyzes the speech probability from the Whisper model to determine
    if actual speech was detected in the audio, helping to prevent empty
    or noise-only transcriptions.

    Args:
        completion: The Whisper API response object containing segments and probabilities

    Returns:
        str or None: Transcribed text if speech was detected, None otherwise
    """
    if completion.segments and len(completion.segments) > 0:
        no_speech_prob = completion.segments[0].get('no_speech_prob', 0)
        print("No speech prob:", no_speech_prob)

        if no_speech_prob > 0.7:
            print("No speech detected")
            return None

        return completion.text.strip()

    return None


def transcribe_audio(client, file_name):
    """
    Transcribe an audio file using the Whisper model via the Groq API.

    Takes an audio file path, sends it to the Whisper speech-to-text service,
    and processes the result to extract the transcribed text.

    Args:
        client: Initialized Groq API client
        file_name: Path to the audio file to transcribe

    Returns:
        str or None: Transcribed text if successful, error message or None if failed
    """
    if file_name is None:
        return None

    try:
        with open(file_name, "rb") as audio_file:
            response = client.audio.transcriptions.with_raw_response.create(
                model="whisper-large-v3-turbo",
                file=("audio.wav", audio_file),
                response_format="verbose_json",
                language="en",
            )
            completion = process_whisper_response(response.parse())

        return completion
    except Exception as e:
        print(f"Error in transcription: {e}")
        return f"Error in transcription: {str(e)}"


def generate_chat_completion(client, history, genre, mood, theme):
    """
    Generate an AI assistant response based on conversation history and song parameters.

    Creates a prompt with the specified musical parameters and sends the entire
    conversation history to the LLM to generate a contextually appropriate response
    that builds on previous exchanges.

    Args:
        client: Initialized Groq API client
        history: List of conversation messages
        genre: Musical genre for context
        mood: Emotional mood for context
        theme: Subject matter/theme for context

    Returns:
        str: Generated assistant response or error message
    """
    messages = []
    system_prompt = f"""You are a creative AI music generator assistant. Help users create song lyrics in the {genre} genre with a {mood} mood about {theme}.
When generating lyrics, create a chorus and at least one verse. Format lyrics clearly with VERSE and CHORUS labels.
Ask if they like the lyrics or want changes. Be conversational, friendly, and creative.
Keep the lyrics appropriate for the selected genre, mood, and theme unless the user specifically requests changes."""

    messages.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    )

    for message in history:
        messages.append(message)

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in generating chat completion: {str(e)}"


theme_gradio = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="blue",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size="md",
    text_size="md",
    spacing_size="md",
)


def start_recording_user(state: AppState):
    """
    Reset the audio recording component for a new user input.

    Args:
        state: Current application state

    Returns:
        None: Indicates the audio component should be reset
    """
    return None


def process_audio(audio: tuple, state: AppState):
    """
    Process recorded audio in real-time during recording.

    This function handles any intermediate processing needed while
    the user is actively recording audio input.

    Args:
        audio: Tuple containing audio data (sample_rate, waveform)
        state: Current application state

    Returns:
        tuple: Processed audio data and updated state
    """
    return audio, state


def update_state_settings(state: AppState, genre_value, mood_value, theme_value):
    """
    Update the application state with new genre, mood, and theme selections.

    Args:
        state: Current application state
        genre_value: Selected musical genre
        mood_value: Selected emotional mood
        theme_value: Selected subject matter/theme

    Returns:
        AppState: Updated application state
    """
    state.genre = genre_value
    state.mood = mood_value
    state.theme = theme_value
    return state


@spaces.GPU(duration=40, progress=gr.Progress(track_tqdm=True))
def response(state: AppState, audio: tuple, genre_value, mood_value, theme_value):
    """
    Process recorded audio and generate a response based on transcription.

    Transcribes user audio input, adds it to the conversation history,
    and generates an assistant response based on the conversation context
    and selected musical parameters.

    Args:
        state: Current application state
        audio: Tuple containing audio data (sample_rate, waveform)
        genre_value: Selected musical genre
        mood_value: Selected emotional mood
        theme_value: Selected subject matter/theme

    Returns:
        tuple: Updated application state and conversation history
    """
    if not audio:
        return AppState(), []

    # Update state with current dropdown values
    state.genre, state.mood, state.theme = genre_value, mood_value, theme_value

    temp_dir = tempfile.gettempdir()
    file_name = os.path.join(temp_dir, f"{xxhash.xxh32(bytes(audio[1])).hexdigest()}.wav")

    sf.write(file_name, audio[1], audio[0], format="wav")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable.")
    client = groq.Client(api_key=api_key)

    # Transcribe the audio file
    transcription = transcribe_audio(client, file_name)
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."

        state.conversation.append({"role": "user", "content": transcription})

        assistant_message = generate_chat_completion(client, state.conversation, state.genre, state.mood, state.theme)

        state.conversation.append({"role": "assistant", "content": assistant_message})

        # print(state.conversation)

        os.remove(file_name)

    return state, state.conversation


# Function to generate music from lyrics
@spaces.GPU(duration=60, progress=gr.Progress(track_tqdm=True))
def generate_music_from_lyrics(state: AppState):
    """
    Generate music based on the lyrics extracted from the conversation.

    Analyzes the conversation history to extract the latest lyrics,
    structures them appropriately, and then uses them to generate
    a musical composition that matches the specified genre, mood, and theme.

    Args:
        state: Current application state containing conversation and parameters

    Returns:
        tuple: Path to generated audio file and status message
    """
    # Extract the final lyrics from the conversation
    # Look for the latest assistant response containing lyrics
    lyrics = ""
    for message in reversed(state.conversation):
        if message["role"] == "assistant" and "verse" in message["content"].lower() and "chorus" in message[
            "content"].lower():
            lyrics = message["content"]
            break

    if not lyrics:
        return None, "ERROR: No lyrics found to generate music. Please create lyrics first be sure that the AI generated at least one CHORUS and VERSE in ONE message."

    try:
        # TODO 1: From the chat history, ask an LLM To generate the complete lyrics with structure output, it could be gcloud, gemma 3 for example
        structured_lyrics = generate_structured_lyrics(
            conversation=state.conversation,
            genre=state.genre,
            mood=state.mood,
            theme=state.theme
        )

        # Format the structured lyrics into a string
        lyrics = format_lyrics_for_yue(structured_lyrics, state.genre, state.mood, state.theme)

        print(lyrics)

        # TODO 2: From the lyrics, generate music using a music generation model (YUE)

        # Save temporary audio file
        # tmp_file = f"/tmp/generated_music_{xxhash.xxh32(lyrics.encode()).hexdigest()}.wav"
        # sf.write(tmp_file, audio_data, sample_rate)

        # return tmp_file, f"Music generated for your {state.genre} song with {state.mood} mood about {state.theme}!"

    except Exception as e:
        error_msg = f"Error generating music: {str(e)}"
        print(error_msg)
        return None, error_msg


# load frontend.js
js = open("frontend.js").read()

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

with gr.Blocks(theme=theme_gradio, js=js) as demo:
    """
    Main Gradio application interface definition.

    Creates a user interface with audio recording capabilities, chat interface,
    music generation controls, and parameter selection dropdowns. The interface
    allows users to:
    1. Select musical parameters (genre, mood, theme)
    2. Record voice input to describe their song ideas
    3. Interact with an AI assistant to refine lyrics
    4. Generate music based on the created lyrics

    The application uses GPU acceleration for AI model inference and provides
    real-time feedback on the generation process.
    """
    gr.HTML("""
    <style>
    #center-text {
        text-align: center;
    }
    </style>
    """)
    gr.Markdown("# 🎵 Hi, this is MiloMusic 🎵", elem_id="center-text")
    gr.Markdown(
        "Just start talking to generate song lyrics. Select genre, mood, and theme below to customize your song.")

    # drop list
    with gr.Row():
        genre = gr.Dropdown(choices=["pop", "rock", "jazz", "hip-hop", "electronic"], value="pop", label="Genre")
        mood = gr.Dropdown(choices=["upbeat", "sad", "energetic", "chill", "romantic"], value="upbeat", label="Mood")
        theme = gr.Dropdown(choices=["love", "breakup", "party", "reflection", "adventure"], value="love",
                            label="Theme")

    with gr.Row():
        input_audio = gr.Audio(
            label="Speak Your Musical Ideas",
            sources=["microphone"],
            type="numpy",
            streaming=False,
            waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        )
    with gr.Row():
        chatbot = gr.Chatbot(label="Creative Conversation", type="messages", height=400)

    with gr.Row():
        generate_btn = gr.Button("🎵 Generate Music from Lyrics", variant="primary")

    with gr.Row():
        music_output = gr.Audio(label="Generated Music", type="filepath")
        generation_status = gr.Textbox(label="Status", placeholder="Click the button above to generate music")

    state = gr.State(value=AppState())

    genre.change(update_state_settings, [state, genre, mood, theme], [state])
    mood.change(update_state_settings, [state, genre, mood, theme], [state])
    theme.change(update_state_settings, [state, genre, mood, theme], [state])

    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )

    respond = input_audio.stop_recording(
        response, [state, input_audio, genre, mood, theme], [state, chatbot]
    )

    restart = respond.then(start_recording_user, [state], [input_audio]).then(
        lambda state: state, state, state, js=js_reset
    )

    generate_btn.click(
        generate_music_from_lyrics,
        [state],
        [music_output, generation_status]
    )

    # Reset button now creates a new AppState with default values
    cancel = gr.Button("New Song", variant="stop")
    cancel.click(
        lambda: (AppState(), gr.Audio(recording=False), None, "Start a new song"),
        None,
        [state, input_audio, music_output, generation_status],
        cancels=[respond, restart],
    )

    gr.Markdown(
        "### How to use:\n1. Select your genre, mood, and theme preferences\n2. Just start talking about your song ideas\n3. The assistant will create lyrics based on your selections\n4. Give feedback to refine the lyrics\n5. When you're happy with the lyrics, click 'Generate Music from Lyrics'\n6. Listen to your generated song!")

if __name__ == "__main__":
    """
    Application entry point when script is executed directly.

    Sets up the environment, validates API keys, and launches the Gradio
    web interface with appropriate system-specific configurations.
    """
    import tempfile

    # Add explicit .env file loading to ensure environment variables are read correctly
    from dotenv import load_dotenv
    import os

    # Specify .env file path - use the .env file in the current directory
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    print(f"Attempting to load environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path)

    # Validate API keys
    keys_valid = validate_api_keys()
    if not keys_valid:
        print("WARNING: One or more API keys failed to load correctly. The application may not function properly!")

    if os.name == "nt":
        demo.launch(allowed_paths=[tempfile.gettempdir()])
    else:
        demo.launch(allowed_paths=["/tmp"])