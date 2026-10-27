# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import asyncio
import os
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ============================================================
# 2. CREATE THE DUCKDUCKGO WEATHER SEARCH TOOL
# ============================================================

def search_weather_news(city: str) -> str:
    """
    Search DuckDuckGo for weather pages and extract current
    weather conditions, temperature, and forecast information.
    """

    # Search for the actual weather conditions in the given city.
    queries = [
        f"{city} current weather temperature today",
        f"{city} weather today temperature rain wind humidity",
        f"{city} current weather conditions feels like temperature",
    ]

    # Store extracted results, errors, and visited URLs.
    page_results = []
    errors = []
    visited_urls = set()

    # Search DuckDuckGo using each query.
    for query in queries:
        try:
            search_results = DDGS().text(
                query=query,
                max_results=8,
            )

            # Inspect each result returned by DuckDuckGo.
            for item in search_results:
                title = item.get("title", "")
                url = item.get("href", item.get("url", ""))

                # Skip results without a title or URL.
                if not title or not url:
                    continue

                # Skip URLs that have already been visited.
                if url in visited_urls:
                    continue

                visited_urls.add(url)

                # Prefer pages that appear to contain weather information.
                title_lower = title.lower()

                if not any(
                    keyword in title_lower
                    for keyword in (
                        "weather",
                        "temperature",
                        "forecast",
                        "current",
                        "conditions",
                    )
                ):
                    continue

                try:
                    # Open the weather webpage found by DuckDuckGo.
                    response = requests.get(
                        url,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 "
                                "(KHTML, like Gecko) "
                                "Chrome/130.0 Safari/537.36"
                            )
                        },
                        timeout=12,
                    )

                    response.raise_for_status()

                    # Parse the webpage HTML.
                    soup = BeautifulSoup(
                        response.text,
                        "html.parser",
                    )

                    # Remove scripts, navigation, and styling.
                    for element in soup(
                        ["script", "style", "nav", "footer", "header"]
                    ):
                        element.decompose()

                    # Extract readable text from the page.
                    page_text = " ".join(soup.stripped_strings)

                    # Skip pages with no readable content.
                    if not page_text:
                        continue

                    # Store the title, URL, and page content.
                    page_results.append(
                        f"Title: {title}\n"
                        f"URL: {url}\n"
                        f"Weather page content: {page_text[:5000]}"
                    )

                except Exception as error:
                    # Record errors when a website cannot be accessed.
                    errors.append(
                        f"{url}: {type(error).__name__}: {error}"
                    )

                # Collect up to three readable weather pages.
                if len(page_results) >= 3:
                    break

        except Exception as error:
            # Record DuckDuckGo search errors.
            errors.append(
                f"DuckDuckGo: {type(error).__name__}: {error}"
            )

        # Stop after collecting enough pages.
        if len(page_results) >= 3:
            break

    # Return extracted weather information to AutoGen.
    if page_results:
        return "\n\n".join(page_results)

    # Print errors to the terminal to help with troubleshooting.
    if errors:
        print("\nWeather search diagnostics:")
        for error in errors:
            print(error)

    # Return a safe fallback if no pages can be read.
    return (
        f"No readable weather information was found for {city}.\n"
        "Current conditions could not be verified from the retrieved pages.\n"
        "Please try again later."
    )


# ============================================================
# 3. CREATE THE MAIN ASYNCHRONOUS FUNCTION
# ============================================================

async def main():

    # Load the API key from the .env file.
    load_dotenv()

    # Read the OpenAI API key securely.
    api_key = os.getenv("OPENAI_API_KEY")

    # Stop if the API key is missing.
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Please check your .env file."
        )

    # Ask the user to enter a city.
    city = input("Which city? ").strip()

    # Validate the city input.
    if not city:
        print("Please enter a city name.")
        return

    # ========================================================
    # 4. CREATE THE OPENAI MODEL CLIENT
    # ========================================================

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        # ====================================================
        # 5. CREATE THE AUTOGEN ASSISTANT AGENT
        # ====================================================

        agent = AssistantAgent(
            name="weather_news",
            model_client=model_client,

            # Register the Python search function as a tool.
            tools=[search_weather_news],

            # Define the agent's role and response format.
            system_message=(
                "You are a Weather Information Agent. "
                "Use the search_weather_news tool to retrieve weather "
                "information for the requested city. "
                "Use only facts supported by the retrieved webpage content. "
                "Do not invent weather conditions or temperatures. "
                "Do not confuse a website description with an actual "
                "weather observation. "
                "Do not claim a reading is current unless the page supports "
                "that claim. "
                "Start your answer with the heading "
                "'Weather News in <City>:' using the requested city name. "
                "After the heading, provide exactly 3 short bullet points. "
                "Include temperature and current conditions when available, "
                "followed by useful forecast, rain, wind, or humidity details. "
                "If information is missing, clearly say it is unavailable. "
                "Do not add any extra introduction or conclusion."
            ),

            # Let the agent interpret the tool results before replying.
            reflect_on_tool_use=True,
        )

        # ====================================================
        # 6. RUN THE AGENT AND DISPLAY ITS ACTIVITY
        # ====================================================

        result = await Console(
            agent.run_stream(
                task=(
                    f"Search DuckDuckGo for current weather conditions "
                    f"and today's forecast in {city}. "
                    f"Title the response 'Weather News in {city}:' "
                    "and follow it with exactly 3 concise bullet points. "
                    "Report actual temperature and conditions only when "
                    "supported by the retrieved page content."
                )
            )
        )

        # Display the final answer separately.
        print("\n========== FINAL ANSWER ==========")
        print(result.messages[-1].content)

    finally:
        # Close the model client even if an error occurs.
        await model_client.close()


# ============================================================
# 7. START THE PROGRAM
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
