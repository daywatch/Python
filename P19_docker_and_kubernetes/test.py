
import requests

# Replace this with the actual URL of your FastAPI /predict endpoint
url = "http://127.0.0.1:8080/predict"

input_text = "today is a [MASK] day!" 

response = requests.post(url, data={"text": input_text})
result = response.json()

print(result)
