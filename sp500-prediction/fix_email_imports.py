#!/usr/bin/env python3
"""
Fix email import case issues in scheduler files
"""

import re
from pathlib import Path

def fix_email_imports(file_path):
    """Fix email import case issues in a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix imports
        content = re.sub(r'from email\.mime\.text import MimeText', 
                        'from email.mime.text import MIMEText', content)
        content = re.sub(r'from email\.mime\.multipart import MimeMultipart', 
                        'from email.mime.multipart import MIMEMultipart', content)
        
        # Fix usage
        content = re.sub(r'\bMimeText\b', 'MIMEText', content)
        content = re.sub(r'\bMimeMultipart\b', 'MIMEMultipart', content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed {file_path.name}")
        
    except Exception as e:
        print(f"✗ Error fixing {file_path.name}: {e}")

def main():
    scheduler_dir = Path("scheduler")
    
    if not scheduler_dir.exists():
        print("Scheduler directory not found")
        return
    
    python_files = list(scheduler_dir.glob("*.py"))
    
    for file_path in python_files:
        fix_email_imports(file_path)
    
    print(f"\nFixed email imports in {len(python_files)} files")

if __name__ == "__main__":
    main()
