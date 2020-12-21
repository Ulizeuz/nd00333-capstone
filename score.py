import requests 
import json  
scoring_uri = '<your web service URI>' 
key = '<your key or token>'  
data = {"data":  [ [xxx, yyy, zzz], [aaa, bbb, ccc] ] } 
input_data = json.dumps(data)
headers = {'Content-Type': 'application/json'} headers['Authorization'] = f'Bearer {key}'
resp = requests.post(scoring_uri, input_data, headers=headers) print(resp.text)
