#!/usr/bin/env python3
"""Return list of names of the home planets of all sentient species
"""

import requests


def sentientPlanets():
    """Return list of names of the home planets of all sentient species
    """
    # URL to fetch species data from Swapi API
    url = "https://swapi-api.alx-tools.com/api/species/"

    # Initialize an empty list to store planet names
    planets = []

    # Loop to handle pagination and fetch data from all pages
    while url:
        # Send a GET request to the Swapi API
        response = requests.get(url)

        # Checks if the response status code is not 200
        if response.status_code != 200:
            break

        # Parse the JSON response into a Python dictionary
        data = response.json()

        # Iterate over each species in the current page of results
        for species in data["results"]:
            homeworld_url = species.get("homeworld")
            if homeworld_url and homeworld_url not in planets:
                try:
                    # Send a GET request to the homeworld URL to fetch planet data
                    homeworld_response = requests.get(homeworld_url)
                    if homeworld_response.status_code == 200:
                        homeworld_data = homeworld_response.json()
                        planets.append(homeworld_data["name"])
                except requests.RequestException:
                    continue

        # Update the URL to fetch the next page of results
        url = data["next"]

    return planets
