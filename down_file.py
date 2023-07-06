import requests
print("downloading with requests")
url = 'http://gateway.ntdjk.com/blade-resource/oss/endpoint/file-view-link?attachId=1420763982380843009'
r = requests.get(url)
with open("demo3.pdf", "wb") as code:
    code.write(r.content)