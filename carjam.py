"""
For handling requests to CarJam API and storing responses.
"""

import requests
import json

response = requests.get('https://www.carjam.co.nz/api/availability/')

args = {
    'plate' : 'KNH93',
    'basic' : 1
}

x = requests.post('https://test.carjam.co.nz/api/car/',
    data= dict(basic=1)
    )

print(x.text)