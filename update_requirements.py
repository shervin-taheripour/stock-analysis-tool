import subprocess
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Target requirements.txt one level up from 'scripts/'
target_path = os.path.join(script_dir, "..", "requirements.txt")
target_path = os.path.abspath(target_path)

print("Writing to:", target_path)

# Run pip freeze and capture output
result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)

# Write to the correct file
with open(target_path, "w", encoding="utf-8") as f:
    f.write(result.stdout)