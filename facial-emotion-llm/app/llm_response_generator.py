import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your .env or system env

def generate_response(emotion):
    prompt = f"The person is showing {emotion} emotion. Respond with a thoughtful and empathetic message."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an empathetic AI companion."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "Sorry, I couldn't generate a response."
