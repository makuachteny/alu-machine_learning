#!/usr/bin/env python3
""" Swapi api returns list of ships and passengers """

import requests


def availableShips(passengerCount):
    """ Return a list of ships that can accommodate a given number of
    passengers.
    Args:
        passengerCount (int): number of passengers.
    Returns:
        list of ships that can accommodate the given number of passengers.
    """
    
    # URL to fetch starships data from Swapi API
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = [] # Initialize an empty list to store ships
    
    # Loop to handle pagination and fetch data from all pages
    while url:
        # Send a GET request to the Swapi API
        response = requests.get(url)
        
        # Checks if the response status code is not 200
        if response.status_code != 200:
            break
        
        # Parse the JSON response into a Python dictionary
        data = response.json()
        
        # Iterate over the each ship in the current page of results
        for ship in data["results"]:
            try:
                # Check if passengers value is numeric or 'unknown'
                passengers = ship["passengers"].strip().replace(",", "")
                if passengers.isnumeric():
                    if int(passengers) >= passengerCount:
                        ships.append(ship["name"])
            except ValueError:
                continue
        url = data["next"]
        
    return ships
