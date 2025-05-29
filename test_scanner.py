import asyncio
import logging
from volatility_scanner import VolatilityScanner, EnhancedMarketRanker, get_supertrend_signals, safe_execute
import os
from dotenv import load_dotenv
import ccxt
import time
import json
import csv
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging (console + file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scanner.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("ScannerTest")

# Strict buy threshold for signals
STRICT_BUY_THRESHOLD = 70

def print_summary_table(results, title="TOP ENHANCED RANKINGS"):
    # Print a simple summary table for the top results
    print(f"\n=== {title} ===")
    headers = ["Rank", "Symbol", "Score", "Tier", "MTF Trend", "Vol%", "Vol24h($)", "Lvg", "Funding%", "OI($)"]
    # Adjust col_widths based on expected content
    col_widths = [6, 18, 8, 28, 18, 8, 12, 5, 10, 12]
    header_str = " | ".join([f"{h:<{col_widths[idx]}}" for idx, h in enumerate(headers)])
    print(header_str)
    print("-" * len(header_str))

    for i, res in enumerate(results, 1):
        symbol = res.get('symbol', '-')
        score = f"{res.get('score', 0.0):.2f}"
        tier = res.get('tier', '-')
        mtf_trend = res.get('mtf_trend', '-')
        
        # Extracting specific metrics from metrics_breakdown for the table
        mb = res.get('metrics_breakdown', {})
        vol_pct_str = mb.get('Volatility (%)', '0.0')
        try: vol_pct = f"{float(vol_pct_str):.2f}" # Already a % string, just ensure formatting
        except: vol_pct = '-'

        vol_24h_str = mb.get('Volume 24h (Quote)', '0')
        try: vol_24h = f"{float(vol_24h_str.replace(',', ''))/1e6:.1f}M" # Format as millions
        except: vol_24h = '-'
        
        leverage = mb.get('Max Leverage', '-')
        
        funding_str = mb.get('Funding Rate (%)', '0.0')
        try: funding = f"{float(funding_str):.4f}"
        except: funding = '-'

        oi_str = mb.get('Open Interest (Quote)', '0')
        try: oi_val = f"{float(oi_str.replace(',', ''))/1e6:.1f}M" # Format as millions
        except: oi_val = '-'

        row_data = [str(i), symbol, score, tier, mtf_trend, vol_pct, vol_24h, leverage, funding, oi_val]
        row_str = " | ".join([f"{str(row_data[j]):<{col_widths[j]}}" for j in range(len(headers))])
        print(row_str)
    print("-" * len(header_str))
    print(f"Displayed top {len(results)} results for: {title}")

def test_scanner(top_n=20, suppress_ohlcv=False, exchange_id='bitget'):
    logger.info("Starting scanner test...")
    start_time = time.time()

    # Initialize VolatilityScanner
    scanner = VolatilityScanner(exchange_id=exchange_id)
    logger.info(f"Fetching USDT-M Futures market data from {exchange_id.capitalize()}...")
    
    # Fetch all swap markets
    try:
        all_markets = scanner.exchange.load_markets()
        swap_markets_raw = { 
            sym: market for sym, market in all_markets.items() 
            if market.get('swap') and market.get('quote') == 'USDT' and market.get('active')
        }
        if not swap_markets_raw:
            logger.error("No active USDT-margined swap markets found.")
            return
        logger.info(f"Found {len(swap_markets_raw)} active USDT-M swap markets.")
    except Exception as e:
        logger.error(f"Error fetching markets: {e}")
        return

    # Categorize markets by leverage
    leverage_categories = {
        "low_leverage (<=20x)": [],
        "mid_leverage (21-50x)": [],
        "high_leverage (>=75x)": [] # Adjusted to >=75x as per user request
    }

    for symbol, market_info_raw in swap_markets_raw.items():
        # Extract max leverage. Bitget stores it in info.maxLeverage or info.maxCrossLeverage usually for swaps
        # It can also be in limits.leverage.max
        max_lev = 1.0 # Default
        if market_info_raw.get('limits') and market_info_raw['limits'].get('leverage'):
            max_lev = market_info_raw['limits']['leverage'].get('max', 1.0)
        
        # Fallback to info fields if not in limits
        if max_lev == 1.0 and market_info_raw.get('info'):
            # Bitget uses strings for leverage in 'info', try to convert
            try:
                max_lev_info = market_info_raw['info'].get('maxLeverage', market_info_raw['info'].get('maxCrossLeverage'))
                if max_lev_info:
                    max_lev = float(max_lev_info)
            except (ValueError, TypeError):
                max_lev = 1.0 # Keep default if conversion fails
        
        if max_lev is None: max_lev = 1.0 # Final ensure it's not None
        
        # Create the market structure expected by scanner.scan_market (which includes 'ticker' later)
        market_data_for_scan = {
            'symbol': symbol,
            'id': market_info_raw.get('id'),
            'base': market_info_raw.get('base'),
            'quote': market_info_raw.get('quote'),
            'active': market_info_raw.get('active'),
            'type': market_info_raw.get('type'),
            'linear': market_info_raw.get('linear'),
            'inverse': market_info_raw.get('inverse'),
            'contractSize': market_info_raw.get('contractSize'),
            'precision': market_info_raw.get('precision'),
            'limits': market_info_raw.get('limits'),
            'info': market_info_raw.get('info'),
            'max_leverage_value': max_lev # Store determined max leverage here
        }

        if max_lev <= 20:
            leverage_categories["low_leverage (<=20x)"].append(market_data_for_scan)
        elif 20 < max_lev <= 50:
            leverage_categories["mid_leverage (21-50x)"].append(market_data_for_scan)
        elif max_lev >= 75:
            leverage_categories["high_leverage (>=75x)"].append(market_data_for_scan)
    
    ranker = EnhancedMarketRanker()
    all_results_categorized = {}

    for category_name, markets_in_category in leverage_categories.items():
        logger.info(f"\n--- Scanning {category_name} ({len(markets_in_category)} pairs) ---")
        if not markets_in_category:
            logger.info(f"No pairs in {category_name}.")
            all_results_categorized[category_name] = []
            continue

        # The VolatilityScanner.scan_specific_markets expects a list of symbol strings or full market dicts
        # We already have market dicts, but they need the 'ticker' field added by scan_specific_markets
        # The VolatilityScanner.scan_market method is what adds ticker and perf data
        
        category_results_scored = []
        processed_symbols_in_category = set()

        for i, market_data in enumerate(markets_in_category):
            symbol = market_data['symbol']
            if symbol in processed_symbols_in_category:
                continue
            
            logger.info(f"Scanning {symbol} ({i+1}/{len(markets_in_category)} in {category_name})...")
            
            # Use safe_execute to run the scanning for each market to add ticker and perf data
            # scan_market will fetch ticker, OHLCV, calculate indicators
            scanned_market_data = safe_execute(scanner.scan_market, market_data, scanner.short_window, scanner.long_window, scanner.rsi_period, scanner.atr_period, scanner.max_workers)
            
            if scanned_market_data and scanned_market_data.get('perf_data'):
                # Ensure 'ticker' is present within scanned_market_data, as ranker expects it
                if 'ticker' not in scanned_market_data and scanned_market_data.get('symbol'):
                    try:
                        ticker_data = scanner.exchange.fetch_ticker(scanned_market_data['symbol'])
                        scanned_market_data['ticker'] = ticker_data
                    except Exception as e:
                        logger.warning(f"Could not fetch ticker for {scanned_market_data['symbol']} during ranker: {e}")
                        scanned_market_data['ticker'] = {'last': 1.0, 'info': {}} # Add dummy ticker
                elif 'ticker' not in scanned_market_data:
                     scanned_market_data['ticker'] = {'last': 1.0, 'info': {}} # Add dummy ticker if symbol also missing

                mtf_signals = get_supertrend_signals(scanned_market_data['symbol'], scanner.exchange)
                scored_data = ranker.score_market(scanned_market_data, mtf_signals, scanned_market_data['perf_data'])
                category_results_scored.append(scored_data)
                processed_symbols_in_category.add(symbol)
            else:
                logger.warning(f"Failed to scan or get perf_data for {symbol}. Skipping.")
        
        # Sort results within the category by score
        category_results_scored.sort(key=lambda x: x['score'], reverse=True)
        all_results_categorized[category_name] = category_results_scored[:top_n]

        if category_results_scored:
            print_summary_table(all_results_categorized[category_name], title=f"TOP {top_n} RESULTS FOR {category_name.upper()}")
            for res_idx, result in enumerate(all_results_categorized[category_name]):
                logger.info(f"\n--- Detailed Enhanced Breakdown for {result['symbol']} (Rank {res_idx + 1} in {category_name}) ---")
                logger.info(result['summary']) # Visual progress bar summary
                
                # Detailed metrics from metrics_breakdown
                mb = result.get('metrics_breakdown', {})
                logger.info("  Metrics Breakdown:")
                for metric_key, metric_value in mb.items():
                    # These are already printed in the tier/score section or are complex
                    if metric_key not in ['Tier', 'Calculated Score', 'MTF Overall Trend', 'MTF Signals', 'summary']:
                        logger.info(f"    {metric_key}: {metric_value}")
                
                logger.info(f"  MTF Overall Trend: {result.get('mtf_trend', 'N/A')}")
                mtf_signals_detail = mb.get('MTF Signals')
                if isinstance(mtf_signals_detail, dict):
                    logger.info("  MTF Signal Details:")
                    for tf_signal, trend_signal in mtf_signals_detail.items():
                        logger.info(f"    {tf_signal} Trend: {trend_signal}")
                
                logger.info(f"  Tier: {result.get('tier', 'N/A')}")
                logger.info(f"  Score: {result.get('score', 0.0):.2f} / {ranker.base_max_score}") # Assuming ranker is accessible or base_max_score is part of result
                # To access ranker.base_max_score, ranker instance must be in scope or base_max_score stored in result
                # For simplicity, let's assume base_max_score is 100 if not directly available in result.
                # The EnhancedMarketRanker now puts 'Calculated Score': f"{final_score:.2f} / {self.base_max_score}" in breakdown.
                # So, we can use that directly if needed, or just show the score.
        else:
            logger.info(f"No scorable results found for {category_name}.")

    # Export all categorized results to JSON and CSV
    try:
        # Flatten results for combined export if needed, or export per category
        combined_results_for_export = []
        for cat_name, cat_res in all_results_categorized.items():
            for r in cat_res:
                r['category'] = cat_name # Add category to each result
                combined_results_for_export.append(r)

        if combined_results_for_export:
            # JSON export (all results)
            json_filename = "scanner_results_categorized.json"
            with open(json_filename, 'w') as f:
                json.dump(combined_results_for_export, f, indent=4)
            logger.info(f"All categorized results exported to {json_filename}")

            # CSV export (flattened metrics_breakdown for easier reading)
            csv_filename = "scanner_results_categorized.csv"
            # Prepare data for CSV: flatten metrics_breakdown
            csv_export_data = []
            for item in combined_results_for_export:
                flat_item = {
                    'symbol': item.get('symbol'),
                    'category': item.get('category'),
                    'score': item.get('score'),
                    'tier': item.get('tier'),
                    'mtf_trend': item.get('mtf_trend')
                }
                # Add all other fields from metrics_breakdown
                if isinstance(item.get('metrics_breakdown'), dict):
                    for mk, mv in item['metrics_breakdown'].items():
                        # Avoid complex objects like dicts for MTF signals in simple CSV
                        if mk == 'MTF Signals' and isinstance(mv, dict):
                            flat_item[mk] = json.dumps(mv) # Store as JSON string
                        else:
                            flat_item[mk] = mv
                csv_export_data.append(flat_item)
            
            if csv_export_data:
                df_results = pd.DataFrame(csv_export_data)
                df_results.to_csv(csv_filename, index=False)
                logger.info(f"All categorized results exported to {csv_filename}")

    except Exception as e:
        logger.error(f"Error exporting results: {e}")

    end_time = time.time()
    logger.info(f"Scanner test finished in {end_time - start_time:.2f} seconds.")
    
    # Minimal OHLCV fetch for a single pair (if not suppressed)
    if not suppress_ohlcv:
        logger.info("\nFetching minimal OHLCV data for BTC/USDT:USDT as a final check...")
        # Ensure API keys are strings, even if empty, for CCXT type consistency
        api_key_val = os.getenv('BITGET_API_KEY', '')
        api_secret_val = os.getenv('BITGET_API_SECRET', '')
        passphrase_val = os.getenv('BITGET_PASSPHRASE', '')

        exchange = ccxt.bitget({
            'apiKey': api_key_val,
            'secret': api_secret_val,
            'password': passphrase_val,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', timeframe='1m', limit=1)
            if ohlcv:
                logger.info(f"BTC/USDT:USDT 1m OHLCV: {ohlcv[0]}")
            else:
                logger.warning("Could not fetch BTC/USDT:USDT OHLCV.")
        except Exception as e:
            logger.error(f"Error fetching BTC/USDT:USDT OHLCV: {e}")

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("scanner.log"),
                            logging.StreamHandler()
                        ])
    # Check Bitget credentials and log warning if missing (moved here for earlier check)
    if not (os.getenv('BITGET_API_KEY') and os.getenv('BITGET_API_SECRET') and os.getenv('BITGET_PASSPHRASE')):
        logger.warning("Bitget API credentials (BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE) are missing or incomplete! Ensure they are in your .env file if private endpoints are needed.")

    test_scanner(top_n=20, suppress_ohlcv=False)
    print("BUSSIED!!!!!") # Acknowledgment as requested

# BUSSIED!!!!!
