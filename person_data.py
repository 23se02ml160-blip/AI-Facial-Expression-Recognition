"""
Person data management using SQLite database.
This module provides a wrapper around the database module for backward compatibility.
"""
from database import load_all_people as load_people, find_person_by_username, register_person
from database import (
    find_person_by_username as find_person,
    register_person,
    load_all_people as load_people,
)

__all__ = ['find_person', 'register_person', 'load_people']
