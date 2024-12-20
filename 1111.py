import openai


openai.api_key='sk-zwoQ1NdOEwO6EeC_PAnhzKVyIEsX65brQYex5gDID6T3BlbkFJQMfDK9IANu2y3S4go2_CN8feCOlcr8qffOvP8PN8kA'

prompt="인상주의 고양이"

response = openai.Image.create(
    model="dall-e-3",
    prompt=prompt,
    n=1,
    size='1024x1024'
)
image_url = response['data'][0]['url']
