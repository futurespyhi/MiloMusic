from pydantic import BaseModel
from typing import List, Optional
import enum


class SectionType(enum.Enum):
    """
    Enumeration of song section types.
    Defines the structural components that can appear in a song's lyrics.
    """
    VERSE = "VERSE"
    CHORUS = "CHORUS"
    BRIDGE = "BRIDGE"
    OUTRO = "OUTRO"
    # PRE_CHORUS = "PRE_CHORUS"


class LyricsSection(BaseModel):
    """
    Represents a single section of lyrics in a song.

    Attributes:
        section_type: The type of section (verse, chorus, etc.)
        content: The actual lyrics text for this section
    """
    section_type: SectionType
    content: str


class SongStructure(BaseModel):
    """
    Represents the complete structure of a song with its title and lyrics sections.

    This model organizes lyrics into a coherent song structure with typed sections.

    Attributes:
        title: The title of the song
        sections: An ordered list of lyric sections that make up the complete song
    """
    title: str
    sections: List[LyricsSection]