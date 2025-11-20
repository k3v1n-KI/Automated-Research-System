#!/usr/bin/env python3
import sys
import re

def replace_p_calls(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace all _p("TYPE", "message") with logger.log("TYPE", "message")
    content = re.sub(r'_p\(([^)]+)\)', r'logger.log(\1)', content)
    
    with open(filename, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)
    replace_p_calls(sys.argv[1])