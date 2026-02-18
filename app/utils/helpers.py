
import os
from pathlib import Path

def update_env_file(updates: dict):
    """
    Update multiple keys in the .env file.
    If a key exists, replace it. If not, append it.
    """
    env_path = Path(".env")
    if not env_path.exists():
        env_path.touch()

    lines = env_path.read_text().splitlines()
    new_lines = []
    processed_keys = set()

    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            new_lines.append(line)
            continue
        
        # Simple parsing key=value
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            
            if key in updates:
                new_value = updates[key]
                # Escape newlines to keep .env file strictly 1 line per key
                if isinstance(new_value, str):
                    new_value = new_value.replace("\r\n", "\\n").replace("\n", "\\n")
                
                # Wrap in quotes if needed or standard behavior
                # For simplicity, we just write key="value" if it was quoted, or just value
                # We'll just standardize on key="value" for updated fields
                new_lines.append(f'{key}="{new_value}"')
                processed_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append new keys
    for key, value in updates.items():
        if key not in processed_keys:
            if isinstance(value, str):
                    value = value.replace("\r\n", "\\n").replace("\n", "\\n")
            new_lines.append(f'{key}="{value}"')

    env_path.write_text("\n".join(new_lines) + "\n")
    
    # Set restrictive permissions (rw-------)
    try:
        os.chmod(env_path, 0o600)
    except Exception:
        # Best effort for permissions
        pass
