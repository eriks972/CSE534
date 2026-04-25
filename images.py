import urllib.request

urls = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg"
]

for i, url in enumerate(urls):
    urllib.request.urlretrieve(url, f"dataset/images/img_{i}.jpg")