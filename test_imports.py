import os
import glob
import importlib.util
import sys

def test_import(file_path):
    """
    Try to import a Python file and report any errors.
    """
    print(f"Testing import of: {file_path}")
    
    # Convert file path to module name
    file_path = file_path.replace('\\', '/')
    rel_path = file_path
    module_name = os.path.splitext(rel_path)[0].replace('/', '.')
    
    try:
        # Try to import the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"  Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"  Error importing {module_name}: {str(e)}")
        return False

def main():
    """
    Test importing Python files.
    """
    # Find all Python files in the specified directories
    directories = ['load/meta_learning', 'load/supervised']
    success_count = 0
    error_count = 0
    
    for directory in directories:
        for file_path in glob.glob(f"{directory}/**/*.py", recursive=True):
            if test_import(file_path):
                success_count += 1
            else:
                error_count += 1
    
    print(f"Summary: {success_count} files imported successfully, {error_count} files failed")

if __name__ == "__main__":
    main() 