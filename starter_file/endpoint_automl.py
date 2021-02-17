import requests
import json

# URL for the web service
scoring_uri = 'http://2660d664-92ad-4de0-8bc2-de786c0de5ca.eastus2.azurecontainer.io/score'

# Input data
data = {
    "data":
    [
        {
            'Red': "24",
            'Green': "250",
            'Blue': "10",
        },
    ],
}

# Convert to JSON 
input_data = json.dumps(data)

# Set header
headers = {'Content-Type': 'application/json'}

# Make request
resp = requests.post(scoring_uri, input_data, headers=headers)

# Display response
print("Response: {}".format(resp.json()))

