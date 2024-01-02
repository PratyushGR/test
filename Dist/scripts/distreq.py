import requests
import json
class distributed():
  def __init__(self, authenticationkey, runid):
    self.auth = 'https://gxiai1odle.execute-api.us-east-1.amazonaws.com/dev/authenticate'
    self.runs = 'https://gxiai1odle.execute-api.us-east-1.amazonaws.com/dev/runs'
    self.keys = authenticationkey
    self.run_id= runid
  
  def upload_gradients(self, gradients):
    data = {"RUNid": self.run_id,"gradients": gradients}
    headers = {'Content-Type': 'application/json','Authorization':self.keys}
    json_data = json.dumps(data)
    response = requests.post(self.runs, data=json_data, headers=headers)

  def get_gradients(self):
    data = {"RUNid": self.run_id}
    headers = {'Content-Type': 'application/json','Authorization':self.keys}
    json_data = json.dumps(data)
    response = requests.get(self.runs, data=json_data, headers=headers)
    return response.json()

