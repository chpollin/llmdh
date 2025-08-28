import requests
from pathlib import Path

def summarize(text):
   """Summarize text using local Mistral model"""
   
   response = requests.post(
       'http://localhost:11434/api/generate',
       json={
           "model": "mistral:7b-instruct-q4_K_M",
           "prompt": f"Summarize this text in 2-3 sentences:\n\n{text}",
           "stream": False
       }
   )
   
   return response.json()['response']

def main():
   """Process all .txt files from the data folder"""
   
   # Get all .txt files from data folder
   txt_files = list(Path("data").glob("*.txt"))
   
   print(f"Found {len(txt_files)} files\n")
   
   # Process each file
   for txt_file in txt_files:
       print(f"File: {txt_file.name}")
       
       # Read file
       text = txt_file.read_text(encoding='utf-8')
       
       # Get summary
       summary = summarize(text)
       
       # Print summary
       print(f"Summary: {summary}\n")
       
       # Save summary
       summary_file = Path("data") / f"{txt_file.stem}_summary.txt"
       summary_file.write_text(summary, encoding='utf-8')
       
       print("-" * 40)

if __name__ == "__main__":
   main()