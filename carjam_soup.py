"""
    Web scraper alternative to API access.
"""

from bs4 import BeautifulSoup
import requests
import re

def carjam_colour(plate: str) -> str:
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
    }
    
    r = requests.get('https://www.carjam.co.nz/car/?plate=' + plate.upper(), headers=headers).text
    soup = BeautifulSoup(r, 'lxml')
    car_colour_html = str(soup.findAll('span', {'class': 'value'})[3])
    car_colour = re.findall( r'>(.*?)<' , car_colour_html)[0]

    return car_colour







