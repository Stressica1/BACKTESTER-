import { create } from 'zustand'

export interface Trade {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  price: number
  timestamp: string
  profit?: number
  status: 'open' | 'closed'
}

export interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  timestamp: string
}

export interface BacktestResult {
  totalTrades: number
  winRate: number
  totalPnL: number
  maxDrawdown: number
  sharpeRatio: number
  startDate: string
  endDate: string
}

export interface ConnectionStatus {
  status: 'connecting' | 'connected' | 'disconnected' | 'error'
  lastHeartbeat?: string
  reconnectAttempts: number
}

interface WebSocketStore {
  // Connection state
  ws: WebSocket | null
  connectionStatus: ConnectionStatus
  
  // Data state
  trades: Trade[]
  marketData: Record<string, MarketData>
  backtestResults: BacktestResult | null
  alerts: Array<{ id: string; message: string; type: 'info' | 'success' | 'warning' | 'error'; timestamp: string }>
  
  // Actions
  connect: (url: string) => void
  disconnect: () => void
  sendMessage: (message: any) => void
  addTrade: (trade: Trade) => void
  updateMarketData: (data: MarketData) => void
  setBacktestResults: (results: BacktestResult) => void
  addAlert: (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void
  removeAlert: (id: string) => void
  clearAlerts: () => void
}

export const useWebSocketStore = create<WebSocketStore>((set, get) => ({
  // Initial state
  ws: null,
  connectionStatus: {
    status: 'disconnected',
    reconnectAttempts: 0,
  },
  trades: [],
  marketData: {},
  backtestResults: null,
  alerts: [],

  // Actions
  connect: (url: string) => {
    const { ws } = get()
    
    // Close existing connection
    if (ws) {
      ws.close()
    }

    set(() => ({
      connectionStatus: {
        ...get().connectionStatus,
        status: 'connecting',
      },
    }))

    try {
      const newWs = new WebSocket(url)
      
      newWs.onopen = () => {
        console.log('WebSocket connected')
        set(() => ({
          ws: newWs,
          connectionStatus: {
            status: 'connected',
            lastHeartbeat: new Date().toISOString(),
            reconnectAttempts: 0,
          },
        }))
        
        // Send initial heartbeat
        newWs.send(JSON.stringify({ type: 'heartbeat', timestamp: new Date().toISOString() }))
        
        // Start heartbeat interval
        const heartbeatInterval = setInterval(() => {
          if (newWs.readyState === WebSocket.OPEN) {
            newWs.send(JSON.stringify({ type: 'heartbeat', timestamp: new Date().toISOString() }))
            set(() => ({
              connectionStatus: {
                ...get().connectionStatus,
                lastHeartbeat: new Date().toISOString(),
              },
            }))
          } else {
            clearInterval(heartbeatInterval)
          }
        }, 30000) // 30 seconds
      }

      newWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          switch (data.type) {
            case 'trade':
              get().addTrade(data.data)
              break
              
            case 'market_data':
              get().updateMarketData(data.data)
              break
              
            case 'backtest_result':
              get().setBacktestResults(data.data)
              break
              
            case 'alert':
              get().addAlert(data.message, data.alertType || 'info')
              break
              
            case 'heartbeat_response':
              set(() => ({
                connectionStatus: {
                  ...get().connectionStatus,
                  lastHeartbeat: new Date().toISOString(),
                },
              }));
              break;
              
            default:
              console.log('Unknown message type:', data.type)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
          get().addAlert('Error parsing server message', 'error')
        }
      }

      newWs.onerror = (error) => {
        console.error('WebSocket error:', error)
        set((state) => ({
          connectionStatus: {
            ...state.connectionStatus,
            status: 'error',
          },
        }))
        get().addAlert('WebSocket connection error', 'error')
      }

      newWs.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason)
        set((state) => ({
          ws: null,
          connectionStatus: {
            status: 'disconnected',
            reconnectAttempts: state.connectionStatus.reconnectAttempts + 1,
          },
        }))
        
        // Auto-reconnect after delay
        if (event.code !== 1000) { // Not a normal closure
          const reconnectDelay = Math.min(1000 * Math.pow(2, get().connectionStatus.reconnectAttempts), 30000)
          get().addAlert(`Connection lost. Reconnecting in ${reconnectDelay / 1000}s...`, 'warning')
          
          setTimeout(() => {
            if (get().connectionStatus.status === 'disconnected') {
              get().connect(url)
            }
          }, reconnectDelay)
        }
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      set((state) => ({
        connectionStatus: {
          ...state.connectionStatus,
          status: 'error',
        },
      }))
      get().addAlert('Failed to establish connection', 'error')
    }
  },

  disconnect: () => {
    const { ws } = get()
    if (ws) {
      ws.close(1000, 'User initiated disconnect')
    }
    set({
      ws: null,
      connectionStatus: {
        status: 'disconnected',
        reconnectAttempts: 0,
      },
    })
  },

  sendMessage: (message: any) => {
    const { ws } = get()
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message))
    } else {
      get().addAlert('Cannot send message: WebSocket not connected', 'error')
    }
  },

  addTrade: (trade: Trade) => {
    set((state) => ({
      trades: [trade, ...state.trades].slice(0, 1000), // Keep last 1000 trades
    }))
  },

  updateMarketData: (data: MarketData) => {
    set((state) => ({
      marketData: {
        ...state.marketData,
        [data.symbol]: data,
      },
    }))
  },

  setBacktestResults: (results: BacktestResult) => {
    set({ backtestResults: results })
    get().addAlert('Backtest completed successfully', 'success')
  },

  addAlert: (message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') => {
    const alert = {
      id: Date.now().toString(),
      message,
      type,
      timestamp: new Date().toISOString(),
    }
    
    set((state) => ({
      alerts: [alert, ...state.alerts].slice(0, 50), // Keep last 50 alerts
    }))

    // Auto-remove alert after 5 seconds (except errors)
    if (type !== 'error') {
      setTimeout(() => {
        get().removeAlert(alert.id)
      }, 5000)
    }
  },

  removeAlert: (id: string) => {
    set((state) => ({
      alerts: state.alerts.filter((alert) => alert.id !== id),
    }))
  },

  clearAlerts: () => {
    set({ alerts: [] })
  },
}))
