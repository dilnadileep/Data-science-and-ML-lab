import requests

def simple_scramper(url):
    response=requests.get(url)
    if response.status_code == 200:
        print("content : ")
        print(response.text)
    else:
        print("failed to fetch : ",response.status_code)
url_to_scraper="https://www.ajce.in"
simple_scramper(url_to_scraper)