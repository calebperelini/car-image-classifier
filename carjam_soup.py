"""Web scraper alternative to API access.
"""

from bs4 import BeautifulSoup
import requests

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
}
r = requests.get('https://www.carjam.co.nz/car/?plate=KNH93', headers=headers).text

soup = BeautifulSoup(r, 'lxml')
car = soup.find(text = 'make')
parent = car.parent
make = parent.find()
print(car)


