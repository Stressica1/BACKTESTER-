import re
import os

def fix_slippage_references():
    """Fix all slippage references in Python files"""
    # List of files to check
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Count of replaced instances
    total_replaced = 0
    
    # Process each file
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace "retrying with market order" with "retrying with market order"
            new_content = re.sub(r'retrying with market order', 'retrying with market order', content)
            
            # Check if there are changes
            if new_content != content:
                # Count replacements
                replacements = content.count('retrying with market order')
                total_replaced += replacements
                
                # Write the fixed content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"Fixed {replacements} slippage references in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Total slippage references fixed: {total_replaced}")

if __name__ == "__main__":
    fix_slippage_references() 