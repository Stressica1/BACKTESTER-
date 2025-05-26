class RiskConfig:
    # Position Sizing
    RISK_PER_TRADE = 0.025  # 2.5% risk per position
    MIN_MARGIN_LEVEL = 0.50  # 50% minimum margin requirement
    LEVERAGE = 35  # 35x leverage
    
    # Take Profit Levels
    TP_LEVELS = [
        {"percentage": 0.02, "size": 0.4},  # 2% TP, 40% of position
        {"percentage": 0.04, "size": 0.3},  # 4% TP, 30% of position
        {"percentage": 0.07, "size": 0.3}   # 7% TP, 30% of position
    ]
    
    # Stop Loss
    STOP_LOSS = 0.025  # 2.5% stop loss
    
    # Entry Strategy
    LADDER_ENTRY = True
    ENTRY_LEVELS = [
        {"percentage": 0.0, "size": 0.4},   # Initial entry, 40% of position
        {"percentage": -0.01, "size": 0.3}, # -1% from initial, 30% of position
        {"percentage": -0.02, "size": 0.3}  # -2% from initial, 30% of position
    ]
    
    # Margin Settings
    MARGIN_MODE = "isolated"  # isolated margin mode
    
    @classmethod
    def calculate_position_size(cls, account_balance: float, entry_price: float) -> float:
        """
        Calculate position size based on risk parameters
        """
        risk_amount = account_balance * cls.RISK_PER_TRADE
        position_size = risk_amount / (entry_price * cls.STOP_LOSS)
        return position_size
    
    @classmethod
    def calculate_tp_prices(cls, entry_price: float) -> list:
        """
        Calculate take profit prices
        """
        return [
            {
                "price": entry_price * (1 + tp["percentage"]),
                "size": tp["size"]
            }
            for tp in cls.TP_LEVELS
        ]
    
    @classmethod
    def calculate_sl_price(cls, entry_price: float) -> float:
        """
        Calculate stop loss price
        """
        return entry_price * (1 - cls.STOP_LOSS)
    
    @classmethod
    def calculate_entry_prices(cls, initial_price: float) -> list:
        """
        Calculate ladder entry prices
        """
        return [
            {
                "price": initial_price * (1 + entry["percentage"]),
                "size": entry["size"]
            }
            for entry in cls.ENTRY_LEVELS
        ]
    
    @classmethod
    def check_margin_safety(cls, margin_level: float) -> bool:
        """
        Check if margin level is safe
        """
        return margin_level >= cls.MIN_MARGIN_LEVEL 