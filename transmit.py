import requests

r = requests.post("http://127.0.0.1:5000/", data={'img': 12524})
print(r.status_code, r.text)
