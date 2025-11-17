import os
import time
from langchain_ollama import ChatOllama

# --- LLM Configuration ---
# Use environment variables for flexibility
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b-llama-distill-q4_K_M")

def get_llm():
    """Initializes and returns the ChatOllama instance."""
    print(f"Connecting to Ollama at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL}...")
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL
    )
    print("Connection successful.")
    return llm

def run_test_connection():
    """A simple test to verify the connection to the LLM."""
    try:
        llm = get_llm()

        # A prompt better suited for a reasoning model
        prompt = "Provide a 3-step plan to build a simple 'to-do list' app."
        print(f"Sending prompt: {prompt}")

        # Start timer
        start_time = time.time()

        # .invoke() works the same, but now it returns a "message" object
        response = llm.invoke(prompt)

        # End timer
        end_time = time.time()

        print("\n--- LLM Response ---")
        # To get the text, we access the '.content' attribute
        print(response.content)
        print(f"-------------------- (took {end_time - start_time:.2f} seconds)")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\n--- Troubleshooting ---")
        print("1. Is Docker running?")
        print("2. Is the Ollama macOS app running?")
        print(f"3. Did you pull the model? `ollama pull {OLLAMA_MODEL}`")
        print(f"4. Is the OLLAMA_BASE_URL correct? Currently: {OLLAMA_BASE_URL}")

if __name__ == "__main__":
    # This script can still be run directly to test the connection.
    # In Phase 2, FastAPI will be the main entry point.
    print("--- Running Connection Test ---")
    run_test_connection()