import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Store your key in env variable

def generate_response(emotion):
    prompt = f"The person looks {emotion}. Respond empathetically in one sentence."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a kind emotional assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']
