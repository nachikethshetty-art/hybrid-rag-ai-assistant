import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

url = "https://api.tavily.com/search"

payload = {
    "api_key": api_key,
    "query": "latest AI news",
    "search_depth": "basic"
}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())