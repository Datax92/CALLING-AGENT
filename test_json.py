import json
import sys

# Test dispatch-rule.json
try:
    with open('dispatch-rule.json', 'r', encoding='utf-8') as f:
        dispatch_rule = json.load(f)
        print("[OK] dispatch-rule.json is valid JSON")
        print(f"Name: {dispatch_rule.get('name')}")
except Exception as e:
    print(f"[ERROR] Error loading dispatch-rule.json: {e}")

# Test inbound-trunk.json
try:
    with open('inbound-trunk.json', 'r', encoding='utf-8') as f:
        inbound_trunk = json.load(f)
        print("[OK] inbound-trunk.json is valid JSON")
        print(f"Trunk name: {inbound_trunk.get('trunk', {}).get('name')}")
except Exception as e:
    print(f"[ERROR] Error loading inbound-trunk.json: {e}")

    # Try to read the file content to see what's wrong
    try:
        with open('inbound-trunk.json', 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"File content: {repr(content)}")
    except Exception as e2:
        print(f"Error reading file content: {e2}")