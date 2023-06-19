import openai

prompt = "I have a beautiful white cat"
openai.api_key = "123"
response = openai.Image.create(
  prompt=Prompt,
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)
