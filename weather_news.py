
import asyncio
import re
import requests

from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Load the OpenAI API key from .env
load_dotenv()


def normalize_text(text: str) -> str:
    """Normalize text for city-name matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def search_weather_news(city: str) -> str:
    """
    Search DuckDuckGo for weather information.
    Reject results that do not mention the requested city.
    Extract relevant webpage text for the AI agent.
    """

    queries = [
        f'"{city}" current weather temperature today',
        f'"{city}" weather forecast today',
        f'"{city}" current weather conditions',
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        )
    }

    city_key = normalize_text(city)
    pages = []
    seen_urls = set()

    weather_keywords = (
        "weather",
        "temperature",
        "forecast",
        "conditions",
    )

    try:
        with DDGS() as ddgs:
            for query in queries:
                results = ddgs.text(query, max_results=8)

                for result in results:
                    title = result.get("title", "")
                    url = result.get("href", "")

                    if not url or url in seen_urls:
                        continue

                    # Verify the requested city appears in the
                    # search result title or URL.
                    result_key = normalize_text(f"{title} {url}")

                    if f" {city_key} " not in f" {result_key} ":
                        continue

                    # Keep weather-related search results only.
                    if not any(
                        word in title.lower()
                        for word in weather_keywords
                    ):
                        continue

                    seen_urls.add(url)

                    try:
                        response = requests.get(
                            url,
                            headers=headers,
                            timeout=10,
                        )
                        response.raise_for_status()

                        # Parse the webpage.
                        soup = BeautifulSoup(
                            response.text,
                            "html.parser",
                        )

                        # Remove irrelevant webpage elements.
                        for element in soup(
                            ["script", "style", "nav", "footer",
                             "header", "aside", "noscript"]
                        ):
                            element.decompose()

                        page_text = soup.get_text(
                            separator=" ",
                            strip=True,
                        )

                        # Verify the requested city is also
                        # mentioned in the extracted webpage text.
                        page_key = normalize_text(page_text)

                        if f" {city_key} " not in f" {page_key} ":
                            continue

                        if len(page_text) < 100:
                            continue

                        pages.append(
                            f"Title: {title}\n"
                            f"URL: {url}\n"
                            f"Weather information:\n"
                            f"{page_text[:4000]}"
                        )

                        # Use a maximum of three verified pages.
                        if len(pages) >= 3:
                            break

                    except requests.RequestException:
                        continue

                if len(pages) >= 3:
                    break

    except Exception as error:
        return f"Weather search failed: {error}"

    if not pages:
        return (
            f"No verified weather information was found for {city}. "
            "The city could not be confirmed in the search results "
            "and webpage text. Check the spelling or try a more "
            "specific city name."
        )

    return "\n\n---\n\n".join(pages)


async def main():
    city = input("Which city? ").strip()

    if not city:
        print("Please enter a city name.")
        return

    # Create the model client.
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
    )

    # Create the AutoGen weather agent.
    agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[search_weather_news],
        reflect_on_tool_use=True,
        system_message=(
            "You are a weather information assistant. "
            "Use the search_weather_news tool to find weather "
            "information for the requested city. "
            "Only report information supported by the tool results. "
            "Never use weather data for a different city. "
            "Never invent temperatures, wind speeds, rain chances, "
            "or other weather conditions. "
            "If no verified information is found, clearly say so "
            "and do not make up weather details. "
            "If information is available, use this exact format:\n"
            "Weather News in <City>:\n"
            "- First key finding.\n"
            "- Second key finding.\n"
            "- Third key finding.\n"
            "Provide exactly three concise bullet points when "
            "verified weather information is available. "
            "If the tool reports that no verified information "
            "was found, explain that instead of inventing three "
            "weather findings."
        ),
    )

    try:
        await Console(
            agent.run_stream(
                task=(
                    f"Find the latest available weather information "
                    f"for {city}. Verify that the information "
                    f"belongs to this city."
                )
            )
        )
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
