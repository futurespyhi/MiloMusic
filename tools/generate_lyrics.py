from google import genai
import os
from typing import List
from schemas.lyrics import LyricsSection, SongStructure
from dotenv import load_dotenv
import json

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()


def generate_structured_lyrics(conversation: List[dict], genre: str, mood: str, theme: str) -> SongStructure:
    """
    Generate structured lyrics using Gemini API based on conversation history and song parameters.

    This function takes a conversation history and song preferences, then uses Google's
    Gemini AI to generate structured song lyrics following a specific format.

    Args:
        conversation: List of conversation messages with 'role' and 'content' keys
        genre: Musical genre for the song (e.g., "pop", "rock")
        mood: Emotional mood for the song (e.g., "romantic", "sad")
        theme: Subject matter or theme of the song (e.g., "love", "friendship")

    Returns:
        SongStructure: A structured representation of the generated song

    Raises:
        ValueError: If no API key is found or if the model returns an empty response
        Exception: For any other errors during API communication or response parsing
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")

    genai_client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    # Convert conversation history into a single prompt string
    conversation_text = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in conversation
    ])

    prompt = f"""Based on the following conversation:

{conversation_text}

Create song lyrics with these parameters:
- Genre: {genre}
- Mood: {mood}
- Theme: {theme}

Generate a complete song with the following structure:
1. A title
2. At least one verse
3. A chorus
4. Optional bridge
5. Optional outro

The output must follow the exact JSON structure with these section types: VERSE, CHORUS, BRIDGE, OUTRO.
"""

    try:
        config = {
            'response_mime_type': 'application/json',
            'response_schema': SongStructure,
            'system_instruction': prompt
        }

        # Pass the prompt as a single string instead of conversation list
        response = genai_client.models.generate_content(
            contents=prompt,
            model=model,
            config=config
        )

        if not response.text:
            raise ValueError("No response generated from the model")

        # Parse the JSON string into a dictionary
        lyrics_data = json.loads(response.text)

        sections = []
        for section in lyrics_data["sections"]:
            sections.append(LyricsSection(
                section_type=section["section_type"],
                content=section["content"]
            ))

        song_structure = SongStructure(
            title=lyrics_data["title"],
            sections=sections,
        )

        return song_structure

    except Exception as e:
        print(f"Error generating lyrics: {str(e)}")
        raise


def format_lyrics(song_structure: SongStructure) -> str:
    """
    Convert the structured lyrics into a formatted string for display.

    This function takes a SongStructure object and creates a human-readable
    text representation with appropriate section headings and spacing.

    Args:
        song_structure: A SongStructure object containing the song title and sections

    Returns:
        str: Formatted lyrics text ready for display
    """
    formatted = f"TITLE: {song_structure.title}\n\n"

    for section in song_structure.sections:
        formatted += f"{section.section_type}:\n{section.content}\n\n"

    return formatted.strip()


def format_lyrics_for_yue(song_structure: SongStructure, genre: str, mood: str, theme: str) -> str:
    """
    Format lyrics in a specialized structure for the YUE music generation system.

    This function converts a SongStructure object into a specific format expected by
    the YUE music generation system, including genre and mood descriptors to guide
    the musical style generation.

    Args:
        song_structure: A SongStructure object containing the song title and sections
        genre: Musical genre that influences the instrumentation descriptors
        mood: Emotional mood that influences the style descriptors
        theme: Subject matter of the song (not directly used in formatting)

    Returns:
        str: Formatted text with appropriate section markers and descriptors for YUE
    """
    genre_descriptors = {
        "pop": "pop vocal clear melodic synthesizer",
        "rock": "rock electric-guitar drums powerful energetic",
        "jazz": "jazz piano smooth saxophone melodic",
        "hip-hop": "rap hip-hop beats vocal rhythmic",
        "electronic": "electronic synthesizer beats modern"
    }

    mood_descriptors = {
        "upbeat": "energetic bright positive",
        "sad": "melancholic emotional soft",
        "energetic": "dynamic powerful strong",
        "chill": "relaxed smooth gentle",
        "romantic": "soft emotional intimate"
    }

    # Create combined genre description
    base_genre = genre_descriptors.get(genre.lower(), "")
    mood_desc = mood_descriptors.get(mood.lower(), "")

    formatted = "Generate music from the given lyrics segment by segment.\n"

    formatted += f"[Genre] {base_genre} {mood_desc} clear vocal\n\n"

    formatted += f"[Title] {song_structure.title}\n\n"

    for section in song_structure.sections:
        section_type = section.section_type.value.lower()

        formatted += f"[{section_type}]\n"

        lines = section.content.strip().split('\n')
        formatted += '\n'.join(line.strip() for line in lines if line.strip())
        formatted += '\n\n'

    return formatted.strip()


# Example usage:
if __name__ == "__main__":
    """
    Test module functionality by generating sample lyrics and displaying them.

    This test code creates a mock conversation about writing a love song,
    generates structured lyrics using the Gemini API, and displays both
    the raw structure and formatted lyrics.
    """
    # Test conversation
    test_conversation = [
        {"role": "user", "content": "I want to write a love song"},
        {"role": "assistant", "content": "I'll help you create a love song. What style are you thinking of?"},
        {"role": "user", "content": "Something romantic and modern"},
        {"role": "assistant",
         "content": "Perfect! Here are the lyrics I've created:\n\nVERSE:\nSoft city lights paint the evening sky\nAs I think about you and I\nEvery moment we've shared feels right\nLike stars aligned in the night\n\nCHORUS:\nThis modern love, it's all we need\nBreaking rules and setting us free\nEvery text, every call, every memory\nMakes this love our reality"}
    ]

    try:
        song = generate_structured_lyrics(
            conversation=test_conversation,
            genre="pop",
            mood="romantic",
            theme="love"
        )

        print("Generated Song Structure:")
        print(song.json(indent=2))

        print("\nFormatted Lyrics:")
        print(format_lyrics(song))

    except Exception as e:
        print(f"Error in test: {str(e)}")