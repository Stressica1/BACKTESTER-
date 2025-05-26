document.addEventListener('DOMContentLoaded', function() {
    // Initialize drawer
    const drawer = document.querySelector('.mdc-drawer');
    const topAppBar = document.querySelector('.mdc-top-app-bar');
    
    if (drawer && topAppBar) {
        const drawerToggle = topAppBar.querySelector('.mdc-top-app-bar__navigation-icon');
        if (drawerToggle) {
            drawerToggle.addEventListener('click', function() {
                drawer.classList.toggle('mdc-drawer--open');
            });
        }
    }
    
    // Initialize side navigation
    document.querySelectorAll('.mdc-list-item').forEach(item => {
        item.addEventListener('click', function() {
            document.querySelectorAll('.mdc-list-item').forEach(i => {
                i.classList.remove('mdc-list-item--activated');
            });
            this.classList.add('mdc-list-item--activated');
            
            // Close drawer on mobile
            if (window.innerWidth <= 1200) {
                drawer.classList.remove('mdc-drawer--open');
            }
            
            // Show corresponding section
            const section = this.getAttribute('data-section');
            if (section) {
                showSection(section);
            }
        });
    });
    
    // Initialize TradingView chart
    initTradingViewWidget();
    
    // Initialize event listeners for trading controls
    initTradingControls();
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Update connection status indicator
    updateConnectionStatus('connecting');
    
    // Initialize sparklines
    initSparklines();
});

function showSection(sectionId) {
    // Implementation needed
    console.log(`Showing section: ${sectionId}`);
}

function initTradingViewWidget() {
    const container = document.getElementById('tradingViewChart');
    if (!container) return;
    
    try {
        // Create TradingView widget
        new TradingView.widget({
            autosize: true,
            symbol: "BINANCE:BTCUSDT",
            interval: "15",
            timezone: "Etc/UTC",
            theme: "dark",
            style: "1",
            locale: "en",
            toolbar_bg: "#131722",
            enable_publishing: false,
            allow_symbol_change: true,
            container_id: "tradingViewChart",
            hide_side_toolbar: false,
            studies: [
                "VWAP@tv-basicstudies"
            ],
            overrides: {
                "mainSeriesProperties.candleStyle.upColor": "#00f5a0",
                "mainSeriesProperties.candleStyle.downColor": "#f15bb5",
                "mainSeriesProperties.candleStyle.wickUpColor": "#00f5a0",
                "mainSeriesProperties.candleStyle.wickDownColor": "#f15bb5",
                "mainSeriesProperties.candleStyle.borderUpColor": "#00f5a0",
                "mainSeriesProperties.candleStyle.borderDownColor": "#f15bb5"
            }
        });
        
        console.log("TradingView widget initialized");
    } catch (error) {
        console.error("Error initializing TradingView widget:", error);
        container.innerHTML = '<div class="error-message">Failed to load chart. Please refresh the page.</div>';
    }
}

function initTradingControls() {
    // Side buttons
    document.querySelectorAll('.side-button').forEach(button => {
        button.addEventListener('click', function() {
            document.querySelectorAll('.side-button').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            
            const side = this.getAttribute('data-side');
            document.getElementById('side').value = side;
            
            // Update execute button class
            const executeButton = document.getElementById('executeTrade');
            if (executeButton) {
                executeButton.classList.remove('buy', 'sell');
                executeButton.classList.add(side);
            }
        });
    });
    
    // Execute trade button
    const executeButton = document.getElementById('executeTrade');
    if (executeButton) {
        executeButton.addEventListener('click', function() {
            const side = document.getElementById('side').value;
            const symbol = document.getElementById('symbol').value;
            const amount = document.getElementById('amount').value;
            const price = document.getElementById('price').value;
            
            if (!side || !symbol || !amount || !price) {
                showAlert('Please fill in all trade parameters', 'warning');
                return;
            }
            
            // Simulate API call
            showAlert(`Processing ${side.toUpperCase()} order...`, 'info');
            
            setTimeout(() => {
                showAlert(`${side.toUpperCase()} order executed: ${amount} ${symbol} at $${price}`, 'success');
                
                // Update active trades
                updateActiveTrades({
                    symbol: symbol,
                    side: side,
                    amount: amount,
                    price: price,
                    timestamp: new Date().toISOString()
                });
            }, 1500);
        });
    }
}

function initWebSocket() {
    // Implementation for WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const wsUrl = `${protocol}${window.location.host}/ws`;
    
    console.log(`Connecting to WebSocket at ${wsUrl}`);
    
    try {
        const socket = new WebSocket(wsUrl);
        
        socket.onopen = function() {
            console.log('WebSocket connected');
            updateConnectionStatus('connected');
            
            // Subscribe to channels
            socket.send(JSON.stringify({
                action: 'subscribe',
                channels: ['market_data', 'trades', 'positions']
            }));
        };
        
        socket.onclose = function() {
            console.log('WebSocket disconnected');
            updateConnectionStatus('disconnected');
            
            // Try to reconnect after 5 seconds
            setTimeout(initWebSocket, 5000);
        };
        
        socket.onerror = function(error) {
            console.error('WebSocket error:', error);
            updateConnectionStatus('disconnected');
        };
        
        socket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        // Store socket in window object for global access
        window.tradingSocket = socket;
    } catch (error) {
        console.error('Failed to connect to WebSocket:', error);
        updateConnectionStatus('disconnected');
        
        // Try to reconnect after 5 seconds
        setTimeout(initWebSocket, 5000);
    }
}

function updateConnectionStatus(status) {
    const statusElement = document.querySelector('.connection-status');
    if (!statusElement) return;
    
    statusElement.className = `connection-status ${status}`;
    
    const statusText = statusElement.querySelector('.connection-status-text');
    if (statusText) {
        switch (status) {
            case 'connected':
                statusText.textContent = 'Connected';
                break;
            case 'connecting':
                statusText.textContent = 'Connecting...';
                break;
            case 'disconnected':
                statusText.textContent = 'Disconnected';
                break;
        }
    }
}

function handleWebSocketMessage(data) {
    // Implementation for handling different message types
    if (!data || !data.type) return;
    
    switch (data.type) {
        case 'market_data':
            updateMarketData(data.data);
            break;
        case 'trade':
            updateTrades(data.data);
            break;
        case 'position':
            updatePositions(data.data);
            break;
        case 'error':
            showAlert(data.message || 'Unknown error', 'danger');
            break;
    }
}

function updateMarketData(data) {
    // Implementation for updating market data display
    console.log('Received market data:', data);
}

function updateTrades(data) {
    // Implementation for updating trades display
    console.log('Received trade data:', data);
}

function updatePositions(data) {
    // Implementation for updating positions display
    console.log('Received position data:', data);
}

function updateActiveTrades(trade) {
    const listContainer = document.querySelector('.active-trades-list');
    if (!listContainer) return;
    
    const tradeItem = document.createElement('div');
    tradeItem.className = `active-trade-item ${trade.side}`;
    tradeItem.innerHTML = `
        <div class="trade-symbol">${trade.symbol}</div>
        <div class="trade-details">
            <span class="trade-size">${trade.amount}</span>
            <span class="trade-price">$${trade.price}</span>
        </div>
        <div class="trade-actions">
            <button class="mini-button close">Close</button>
        </div>
    `;
    
    listContainer.prepend(tradeItem);
    
    // Add close handler
    const closeButton = tradeItem.querySelector('.close');
    if (closeButton) {
        closeButton.addEventListener('click', function() {
            tradeItem.style.animation = 'slide-out-right 0.3s forwards';
            setTimeout(() => {
                tradeItem.remove();
                showAlert(`${trade.symbol} position closed`, 'success');
            }, 300);
        });
    }
}

function initSparklines() {
    document.querySelectorAll('.stats-sparkline').forEach(element => {
        const values = element.getAttribute('data-values').split(',').map(Number);
        const width = element.clientWidth;
        const height = element.clientHeight;
        
        if (width <= 0 || height <= 0 || !values.length) return;
        
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1;
        
        const points = values.map((value, i) => {
            const x = (i / (values.length - 1)) * width;
            const y = height - ((value - min) / range * height);
            return `${i === 0 ? 'M' : 'L'}${x},${y}`;
        }).join(' ');
        
        element.innerHTML = `
            <svg width="${width}" height="${height}">
                <path d="${points}" stroke="currentColor" fill="none" stroke-width="2" />
            </svg>
        `;
    });
}

function showAlert(message, type = 'info') {
    const alertWidget = document.getElementById('alertWidget');
    if (!alertWidget) return;
    
    const alertId = 'alert-' + Date.now();
    const alertTitle = {
        'info': 'Information',
        'success': 'Success',
        'warning': 'Warning',
        'danger': 'Alert'
    }[type] || 'Notification';
    
    const alertItem = document.createElement('div');
    alertItem.className = `alert-item ${type}`;
    alertItem.id = alertId;
    alertItem.innerHTML = `
        <div class="alert-header">
            <div class="alert-title">${alertTitle}</div>
            <button class="alert-close">&times;</button>
        </div>
        <div class="alert-content">${message}</div>
        <div class="alert-time">${new Date().toLocaleTimeString()}</div>
    `;
    
    alertWidget.appendChild(alertItem);
    
    // Set up close handler
    alertItem.querySelector('.alert-close').addEventListener('click', () => {
        alertItem.style.animation = 'slide-out-right 0.3s forwards';
        setTimeout(() => {
            alertItem.remove();
        }, 300);
    });
    
    // Auto-remove after 8 seconds
    setTimeout(() => {
        if (document.getElementById(alertId)) {
            alertItem.style.animation = 'slide-out-right 0.3s forwards';
            setTimeout(() => {
                alertItem.remove();
            }, 300);
        }
    }, 8000);
}
