# The fix is here: We import 'ChatOllama' instead of 'Ollama'
from langchain_ollama import ChatOllama

def main():
    # Use the new, specialized model name
    MODEL_NAME = "deepseek-r1:8b-llama-distill-q4_K_M"
    
    print(f"Connecting to Ollama and loading model: {MODEL_NAME}...")
    
    # IMPORTANT: We use the Docker service name 'ollama', not 'localhost'
    try:
        # The fix is also here: We instantiate 'ChatOllama'
        llm = ChatOllama(
            base_url="http://host.docker.internal:11434",
            model=MODEL_NAME
        )
        
        print("Connection successful.")
        
        # A prompt better suited for a reasoning model
        prompt = "Provide a 3-step plan to build a simple 'to-do list' app."
        print(f"Sending prompt: {prompt}")
        
        # .invoke() works the same, but now it returns a "message" object
        response = llm.invoke(prompt)
        
        print("\n--- LLM Response ---")
        # To get the text, we access the '.content' attribute
        print(response.content)
        print("--------------------")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\n--- Troubleshooting ---")
        print("1. Is Docker running?")
        print("2. Did 'docker-compose up -d' run successfully?")
        print(f"3. Did 'docker exec -it ollama_service ollama pull {MODEL_NAME}' complete?")
        print("4. Is the 'base_url' set to 'http://ollama:11434' (the service name)?")

if __name__ == "__main__":
    main() 
    # The fix is here: We import 'ChatOllama' instead of 'Ollama' from langchain_ollama import ChatOllama