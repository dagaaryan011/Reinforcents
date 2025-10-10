# diagnose_file.py
import json
from pathlib import Path

# This path MUST be identical to the one in your app.py and dashboard_updater.py
from config  import DATA_FILE 

print(f"--- 🕵️‍♂️ Diagnosing file: {DATA_FILE} ---")

if not DATA_FILE.exists():
    print("\nRESULT: ❌ The file does not exist.")
    print("This means main.py is not creating it.")
else:
    try:
        with DATA_FILE.open('r') as f:
            # Read the raw text first to check if it's empty
            raw_content = f.read()
            if not raw_content.strip():
                print("\nRESULT: ⚠️ The file is empty.")
                exit()
            
            # Go back to the start of the file to load json
            f.seek(0)
            content = json.load(f)
            
        print(f"\nSuccessfully loaded JSON content.")
        print(f"TYPE of content: {type(content)}")
        print("\n--- Content Sample (first 500 chars) ---")
        print(str(content)[:500])
        print("----------------------------------------")

        if isinstance(content, dict):
            print("\nRESULT: ✅ The file contains a DICTIONARY, which is correct.")
            print("If you still get the error, the problem is very unusual.")
        elif isinstance(content, list):
            print("\nRESULT: ❌ The file contains a LIST. This is the source of the error.")
            print("This PROVES that your main.py is still running an OLD version of dashboard_updater.py.")
        else:
            print(f"\nRESULT: ⚠️ The file contains an unexpected type: {type(content)}")

    except json.JSONDecodeError:
        print("\nRESULT: ❌ The file is corrupted or not valid JSON.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")