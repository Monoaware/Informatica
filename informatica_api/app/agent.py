import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import tool

from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent


load_dotenv()


llm = ChatGoogleGenerativeAI(

    model="gemini-2.0-flash",

    google_api_key=os.environ.get("GOOGLE_API_KEY")

)


@tool

def get_trending_topics(platform: str) -> str:

    """Get trending topics for a given social media platform.

    Use this when the user asks about what content is currently popular."""

    # Your actual implementation here

    return f"Trending topics for {platform}: AI art, productivity tips, behind-the-scenes content"


system_prompt = """You are AlgoRhythm, a content creator assistant.

You help creators plan posts, choose hashtags, and identify trending topics.

Use your tools when the user asks about trends or platform-specific content."""


agent = create_react_agent(

    model=llm,

    tools=[get_trending_topics],

    prompt=system_prompt

)