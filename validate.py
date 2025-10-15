#!/usr/bin/env python3
"""
Validate Python code structure without requiring dependencies.
"""
import ast
import sys

files_to_check = [
    'main.py',
    'model.py', 
    'config.py',
    'test_service.py',
    'example_client.py'
]

print('Validating Python syntax...')
print('=' * 50)

all_valid = True
for filename in files_to_check:
    try:
        with open(filename, 'r') as f:
            ast.parse(f.read())
        print(f'✓ {filename} - valid syntax')
    except SyntaxError as e:
        print(f'✗ {filename} - syntax error: {e}')
        all_valid = False
    except FileNotFoundError:
        print(f'✗ {filename} - file not found')
        all_valid = False

print('=' * 50)
if all_valid:
    print('All Python files have valid syntax! ✓')
    sys.exit(0)
else:
    print('Some files have errors! ✗')
    sys.exit(1)
