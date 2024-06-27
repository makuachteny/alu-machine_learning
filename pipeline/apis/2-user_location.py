#!/usr/bin/env python3

"""
Script to print the location of a specific GitHub user.
"""

import requests
import sys
import time


def get_user_location(url):
    """Fetches and prints the location of a specific GitHub user."""
    try:
        response = requests.get(url)

        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get("location", "Location not available")
            print(location)
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            rate_limit_reset = int(response.headers.get('X-Ratelimit-Reset'))
            current_time = int(time.time())
            diff = (rate_limit_reset - current_time) // 60
            print(f"Reset in {diff} min")
        else:
            print(f"Error: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    api_url = f"https://api.github.com/users/holbertonschool"
    get_user_location(api_url)