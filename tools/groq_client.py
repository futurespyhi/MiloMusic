import os
import groq
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

"""
Initialize Groq API client for large language model access.

This module sets up a connection to the Groq API service, which provides
access to fast large language models. The API key is securely retrieved
from environment variables rather than being hardcoded.

Environment Variables:
    GROQ_API_KEY: Personal API key for Groq service authentication

Raises:
    ValueError: If the GROQ_API_KEY environment variable is not set

Usage:
    Import this module to get access to a pre-configured Groq client
    that can be used for making API requests to Groq's language models.
"""

# Initialize Groq client securely
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")
client = groq.Client(api_key=api_key)