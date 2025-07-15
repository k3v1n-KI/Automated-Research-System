import requests

url = "https://mcp-read.ludwig.link/query"
payload = {
    "tool": "read_website_fast",
    "input": "https://www.oha.com/membership/oha-members"
}

response = requests.post(url, json=payload)
cleaned_text = response.json().get("output")
print(cleaned_text[:500])  # Preview of the extracted content
