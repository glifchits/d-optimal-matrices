import json
import requests


def post_to_google_sheets(d):
    URL = 'http://sheetsu.com/apis/5c009ac6'
    # https://docs.google.com/spreadsheets/d/1GJIBza3e31SvU07aLT2GV7kYlyI1679koes1LTJleTI
    data = json.dumps(d)
    headers = { 'Content-Type': 'application/json' }
    r = requests.post(URL, data=data, headers=headers)
    resp = json.loads(r.text)
    if resp['status'] != 201:
        raise Error(resp['error'])

