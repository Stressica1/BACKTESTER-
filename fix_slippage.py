import re

with open('supertrend_pullback_live.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the log message for price deviation errors
content = content.replace('logger.warning(f"⚠️ Price deviation error (50067) - retrying with market order to 15.0%', 
                         'logger.warning(f"⚠️ Price deviation error (50067) - retrying with market order')

# Replace any other references to slippage in the log messages
content = content.replace('retrying with market order', 'retrying with market order')

# Write the fixed content back to the file
with open('supertrend_pullback_live.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Now fix the error manager
try:
    with open('bitget_error_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace slippage logging in the error manager
    content = content.replace('retrying with market order', 'retrying with market order')
    
    # Write the fixed content back
    with open('bitget_error_manager.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Successfully removed slippage references from bitget_error_manager.py")
except Exception as e:
    print(f"Error fixing bitget_error_manager.py: {e}")

print("Completed fixing all slippage references!")
