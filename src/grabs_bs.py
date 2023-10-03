


from bs4 import BeautifulSoup

import requests

r = requests.get("https://www.ageofempires.com/stats/?profileId=199325&game=age2&matchType=3")
data = r.text
soup = BeautifulSoup(data)
adf = 5
soup = BeautifulSoup(r.content, "html.parser")
aa = soup.find_all('script')