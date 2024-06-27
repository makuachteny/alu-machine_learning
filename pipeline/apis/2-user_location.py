#!/usr/bin/env python3

"""
Script to print the location of a specific GitHub user.
"""

import requests
import sys
import time


def get_user_location(url):
    """
    Fetches and prints the location of a specific GitHub user.
    
    Parameters:
        url (str): The GitHub API URL for the user.

    Prints:
        The location of the user if available, otherwise appropriate messages
        based on the HTTP response status.
    """
    try:
        # Send a GET request to the provided URL
        response = requests.get(url)

        # Check the status code of the response
        if response.status_code == 200:
            # Parse the JSON response
            user_data = response.json()
            # Get the location from the response, or a default message if not available
            location = user_data.get("location", "Location not available")
            print(location)
        elif response.status_code == 404:
            # User not found
            print("Not found")
        elif response.status_code == 403:
            # Rate limit exceeded
            rate_limit_reset = int(
                response.headers.get('X-Ratelimit-Reset', 0))
            current_time = int(time.time())
            diff = (rate_limit_reset - current_time) // 60
            print(f"Reset in {diff} min")
        else:
            # Other errors
            print(f"Error: {response.status_code}")
    except requests.RequestException as e:
        # Handle request exceptions
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check if the URL argument is provided
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
    else:
        # Get the API URL from command-line arguments
        api_url = sys.argv[1]
        # Fetch and print the user's location
        get_user_location(api_url)
