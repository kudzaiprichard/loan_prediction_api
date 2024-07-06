# location_handler.py

# Dictionary mapping location strings to integers
location_dict = {
    "beitbridge": 1,
    "bulawayo": 2,
    "chimanimani": 3,
    "chipinge": 4,
    "chiredzi": 5,
    "chivhu": 6,
    "gokwe": 7,
    "gweru": 8,
    "harare": 9,
    "hwange": 10,
    "kadoma": 11,
    "kariba": 12,
    "karoi": 13,
    "kwekwe": 14,
    "marondera": 15,
    "masvingo": 16,
    "mutare": 17,
    "nyanga": 18,
    "plumtree": 19,
    "redcliff": 20,
    "rusape": 21,
    "shurugwi": 22,
    "victoria falls": 23,
    "zvishavane": 24
}

# Default value if location is not in the dictionary
default_location = 25


def get_location_id(location_str):
    """Get location ID from dictionary or return default_location if not found."""
    location_str = location_str.lower()  # Convert to lowercase for case insensitivity
    return location_dict.get(location_str, default_location)
