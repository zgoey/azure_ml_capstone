import requests
import json

# URL for the web service
scoring_uri = 'http://1e75aeda-c4df-4746-98b9-4e209326713e.eastus2.azurecontainer.io/score'

# Input
data = {"data": [[245, 250, 10]]}

# Convert to JSON 
input_data = json.dumps(data)

# Set header
headers = {'Content-Type': 'application/json'}

# Make request
resp = requests.post(scoring_uri, input_data, headers=headers)

# Display raw response
print("Raw response: {}".format(resp.json()))

# Display decoded response
label_dict = {0: 'Black', 1: 'Blue', 2: 'Brown', 3: 'Green', 4: 'Grey', 5: 'Orange', 6: 'Pink', 7: 'Purple', 8: 'Red', 9: 'White', 10: 'Yellow'}
print("Decoded response: {}".format([label_dict[r] for r in resp.json()]))