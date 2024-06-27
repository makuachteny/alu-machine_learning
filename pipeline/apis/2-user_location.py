#!/usr/bin/env python3
""" Github api returns user location """

import requests
import sys
import time


if __name__ == "__main__":
    res = requests.get('https://api.github.com/users/' + sys.argv[1])

    if res.status_code == 403:
        rate_limit = int(res.headers.get('X-RateLimit-Remaining'))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print("Reset in {} min".format(diff))
        # get remaining rate limit
    elif res.status_code == 404:
        print("User not found")
    else:
        print(res.json().get('location'))
