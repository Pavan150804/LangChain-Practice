from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

# Load API key from .env
load_dotenv()

# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Define output schema
class Review(TypedDict):
    key_themes: Annotated[
        list[str],
        "Write down all key themes discussed in the review"
    ]

    summary: Annotated[
        str,
        "A brief summary of the review"
    ]

    sentiment: Annotated[
        Literal["positive", "negative"],
        "Overall sentiment of the review"
    ]

    strength: Annotated[
        Optional[list[str]],
        "List all strengths mentioned in the review"
    ]

    weakness: Annotated[
        Optional[list[str]],
        "List all weaknesses mentioned in the review"
    ]

    name: Annotated[
        Optional[str],
        "Name of the reviewer"
    ]

# Convert normal model into structured model
structured_model = model.with_structured_output(Review)

# Input review
review_text = """
I watched the movie Eternal Horizons last night, and it left me
thinking about life in an entirely new way.

The storytelling is slow but purposeful, allowing each character
arc to breathe.

The visuals are breathtaking, with wide shots of desolate
landscapes mixed with intimate emotional moments.

The soundtrack is haunting and perfectly complements the movie's tone.

The performances, especially by the lead actress, feel raw and authentic.

However, the movie is definitely not for everyone.
The pacing is extremely slow, and some scenes drag on longer than necessary.

The ending is deliberately ambiguous, which might frustrate viewers
who prefer clear resolutions.

Review by Arjun Rao
"""

# Extract structured data
result = structured_model.invoke(review_text)

# Print output
print(result)