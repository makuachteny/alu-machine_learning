#!/usr/bin/env python3

"""
Script to print the location of a specific GitHub user.
"""

import requests
import sys
import time


def get_user_location(api_url):
    """
    Fetches location of a GitHub user.

    Args:
        api_url (str): The full API URL for the GitHub user.

    Prints:
        - The location of the user if available.
        - "Reset in X min" if rate limit exceeded.
        - "Not found" if the user does not exist.
        - Error messages for other HTTP status codes or request exceptions.
    """
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get("location", "Location not available")
            print(location)
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            rate_limit_reset = int(
                response.headers.get('X-Ratelimit-Reset', 0))
            current_time = int(time.time())
            diff = (rate_limit_reset - current_time) // 60
            print("Reset in {} min".format(diff))
        else:
            print("Error: {}".format(response.status_code))
    except requests.RequestException as e:
        print("Error: {}".format(e))


if __name__ == "__main__":
    """
    Main execution block for fetching GitHub user location.

    Usage:
        ./2-user_location.py <GitHub API URL>
    """
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
    else:
        api_url = sys.argv[1]
        get_user_location(api_url)
