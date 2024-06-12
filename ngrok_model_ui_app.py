import warnings
import config as cfg
import streamlit as st
from pyngrok import ngrok
import subprocess

warnings.filterwarnings("ignore")


# Define the Streamlit app function to run in the background
def run_streamlit():
    subprocess.Popen(
        ["streamlit", "run", "app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


# Start ngrok with the specified domain
def start_ngrok():
    ngrok.kill()  # Kill any existing ngrok tunnels
    ngrok.set_auth_token(cfg.NGROK_AUTH_TOKEN)  # Set up ngrok auth token
    public_url = ngrok.connect(
        addr="localhost:8501",
        options={
            "bind_tls": True,
            "hostname": "creative-personally-sunbird.ngrok-free.app",
        },
    )
    print(f"Public URL: {public_url.public_url} - Open this link in your browser")


if __name__ == "__main__":
    start_ngrok()
    run_streamlit()
