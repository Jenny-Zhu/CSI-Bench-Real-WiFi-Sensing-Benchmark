import os
import glob

def fix_null_bytes(file_path):
    """Remove null bytes from a file and save it with UTF-8 encoding"""
    try:
        # Try different encodings, starting with utf-8 and falling back to latin-1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Binary mode as last resort
                with open(file_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')

        # Check if there are null bytes
        if '\0' in content:
            print(f"Fixing null bytes in {file_path}")
            content = content.replace('\0', '')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    # Find all Python files in the project
    python_files = glob.glob('**/*.py', recursive=True)
    
    fixed_count = 0
    for file_path in python_files:
        if fix_null_bytes(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files with null bytes")
    
    # Check specific problematic imports directories
    problematic_dirs = [
        'load/meta_learning',
        'load/supervised',
        'engine/meta_learning',
        'engine/supervised'
    ]
    
    for dir_path in problematic_dirs:
        if os.path.exists(dir_path):
            print(f"\nChecking directory: {dir_path}")
            init_file = os.path.join(dir_path, '__init__.py')
            
            # Fix or create the __init__.py file
            if os.path.exists(init_file):
                fix_null_bytes(init_file)
                print(f"Fixed {init_file}")
            else:
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write('# Module initialization\n')
                print(f"Created {init_file}")

if __name__ == "__main__":
    main() 