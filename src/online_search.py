# Standard logging module for outputting log messages
import logging

# Type hinting to specify return types for better readability and editor support
from typing import List

# Import the Google search function from the googlesearch library
from googlesearch import search

# Selenium for browser automation
from selenium import webdriver

# Automatically downloads the appropriate ChromeDriver version
from webdriver_manager.chrome import ChromeDriverManager

# Handle page load timeout exceptions
from selenium.common.exceptions import TimeoutException

# Import Chrome options and services for WebDriver configuration
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService

# Importing modules for locating elements and managing waits
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Importing custom summarization model
from .model_tokenizer import text_summarization

# Set up logging configuration with INFO level to see logs in the console
logging.basicConfig(level=logging.INFO)


def get_website_text(
    url: str, 
    chrome_driver_path: str = None, 
    max_wait_time: int = 10
) -> str:
    """
    Extracts text content from a given website using Selenium.
    """

    # Set up Chrome browser options to run headless (no UI) and disable non-text content for speed
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")                               # Run browser without a UI
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")     # Disable images
    chrome_options.add_argument("--disable-javascript")                     # Disable JS to speed up page load
    chrome_options.add_argument("--disable-plugins")                        # Disable plugins to reduce load time

    # Configure the Chrome driver service. Use user-specified path or install automatically.
    chrome_service = (
        ChromeService(executable_path=chrome_driver_path)
        if chrome_driver_path
        else None
    )
    chrome_service = chrome_service or ChromeService(ChromeDriverManager().install())

    # Open a Chrome browser session
    with webdriver.Chrome(service=chrome_service, options=chrome_options) as driver:  # type: WebDriver
        # Set the maximum wait time for elements to load
        driver.implicitly_wait(max_wait_time)

        # Navigate to the provided URL
        logging.info("URL is: %s", url)
        driver.get(url)

        try:
            # Explicitly wait for the presence of the body element on the page
            WebDriverWait(driver, max_wait_time).until(
                EC.presence_of_element_located((By.XPATH, "/html/body"))
            )
        except TimeoutException:
            # Log a warning if the page didn't load in time
            logging.warning("Page load timed out for URL: %s", url)

        # Extract and return the visible text from the entire body of the page
        page_text = driver.find_element(By.XPATH, "/html/body").text  # type: WebElement
        return page_text


def google_search(
    query: str, search_time: str, num_results: int = 3, lang: str = "en"
) -> List[str]:
    """
    Perform a Google search and return URLs for the top search results.
    """

    # Define a maximum length for the user's query to ensure relevance and precision
    max_user_query_len = 10

    # If the query has too many words, summarize it for better search relevance
    if len(query.split()) > max_user_query_len:
        logging.info("Query to send before sending to Google:\n%s", query)
        query = summarize_text(query)   # Summarize if it's too long
        logging.info("Query to Google after summarization: %s", query)

    # Mapping of search time windows to Google's TBS (Time-Based Search) codes
    dic_time = {
        "All": "a",
        "Year": "y",
        "Month": "m",
        "Week": "w",
        "Day": "d",
        "Hour": "h",
    }

    search_results = []
    # Perform the Google search using the modified query and time range
    for result in search(
        query,
        num=num_results,                    # Number of results to return
        stop=num_results,                   # Stop after this many results
        pause=3,                            # Delay between HTTP requests to avoid rate limiting
        lang=lang,                          # Language filter for search results
        tld="com",                          # Top-level domain (e.g., google.com)
        tbs=f"qdr:{dic_time[search_time]}", # Apply time-based search filter
    ):
        search_results.append(result)       # Add each result URL to the list

    return search_results


def summarize_text(
    text: str, max_length: int = 25, min_length: int = 3, do_sample: bool = False
) -> str:
    """
    Summarizes input text to make it more concise, especially useful for long search queries.
    """

    # Load the summarization model from model_tokenizer
    model, _ = text_summarization()

    # Generate summary using the model and parameters
    summary = model(
        text, max_length=max_length, min_length=min_length, do_sample=do_sample
    )

    # Return the summarized text content
    return summary[0]["summary_text"]
