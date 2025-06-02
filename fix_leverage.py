with open('supertrend_pullback_live.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the get_optimal_slippage function
content = content.replace('''    def get_optimal_slippage(self, symbol, retry_count=0):
        """Return a fixed slippage value"""
        # Fixed slippage regardless of symbol or retry count
        return 5.0  # Fixed 5% slippage

''', '')

# Fix the set_leverage function - remove result =
content = content.replace('            result = self.exchange.set_leverage(leverage, symbol=symbol, params=params)',
                         '            self.exchange.set_leverage(leverage, symbol=symbol, params=params)')

content = content.replace('                    result = self.exchange.set_leverage(leverage, symbol=symbol, params=params)',
                         '                    self.exchange.set_leverage(leverage, symbol=symbol, params=params)')

content = content.replace('                        result = self.exchange.set_leverage(adjusted_leverage, symbol=symbol, params=params)',
                         '                        self.exchange.set_leverage(adjusted_leverage, symbol=symbol, params=params)')

# Remove any references to "retrying with market order" in error handling
content = content.replace('WARNING - ⚠️ Price deviation error (50067) - retrying with market order', 
                         'WARNING - ⚠️ Price deviation error (50067) - retrying with market order')

# Write the fixed content back
with open('supertrend_pullback_live.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied successfully!") 