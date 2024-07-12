#!/usr/bin/env python3
""" 
This script is used to insert a new school into a MongoDB collection and print the details of all schools in the collection.

Usage:
    python 31-insert_school.py

Dependencies:
    - pymongo

Functions:
    - insert_school(collection, name, address): Inserts a new school document into the specified collection with the given name and address.
    - list_all(collection): Retrieves all school documents from the specified collection.

Example:
    $ python 31-insert_school.py
    New school created: 1234567890
    [1234567890] UCSF 505 Parnassus Ave
    [0987654321] Stanford University 450 Serra Mall
    ...
"""

from pymongo import MongoClient
list_all = __import__('30-all').list_all
insert_school = __import__('31-insert_school').insert_school

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school
    new_school_id = insert_school(
        school_collection, name="UCSF", address="505 Parnassus Ave")
    print("New school created: {}".format(new_school_id))

    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'),
              school.get('name'), school.get('address', "")))
