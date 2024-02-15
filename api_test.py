import requests

# headers = {"Cookie": "access-token=njoKuRFCd5KkPYPi2FKMbg"}
# headers = {"Cookie": "access-token-unsecure=6_q0Y0zEkx1BOOM3Fxi9NA"}
headers = {"Cookie": "access-token-unsecure=trTd-1ncqzNpzUw1PxCaBA"}
youtube_link = "https://www.youtube.com/watch?v=ItROv_PnB4o&ab_channel=PeterPannekoek"
project_name = "peter-pannekoek-2"
soure_lang = "nl"
response = requests.post(
    "http://192.168.2.2:8000/run/end2end",
    headers=headers,
    json={
        "data": [
            {"name": "",
             "data": ""},
            youtube_link,
            project_name,
            soure_lang,
            "EN-US",
            ["Demo Run", "Save speakers"],
        ]
    }).json()

# data = response["data"]
print(response)
