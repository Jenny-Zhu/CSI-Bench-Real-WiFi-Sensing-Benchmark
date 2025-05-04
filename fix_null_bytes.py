import os
import glob

def fix_file(file_path):
    """
    Read a file in binary mode, remove null bytes and write it back.
    """
    print(f"Checking file: {file_path}")
    
    # Read the file in binary mode
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Check if it contains null bytes
    if b'\x00' in content:
        print(f"  Fixing null bytes in {file_path}")
        
        # Remove null bytes
        cleaned_content = content.replace(b'\x00', b'')
        
        # Write the file back
        with open(file_path, 'wb') as f:
            f.write(cleaned_content)
        
        print(f"  Fixed {len(content) - len(cleaned_content)} null bytes")
        return True
    
    return False

def main():
    """
    Fix null bytes in Python files.
    """
    # Find all Python files in the specified directories
    directories = ['load/meta_learning', 'load/supervised']
    fixed_count = 0
    
    for directory in directories:
        for file_path in glob.glob(f"{directory}/**/*.py", recursive=True):
            if fix_file(file_path):
                fixed_count += 1
    
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main() 