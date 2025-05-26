import { useEffect, useRef, useState } from 'react'
import { Card, CardContent, Box, Typography, ToggleButtonGroup, ToggleButton } from '@mui/material'
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts'
import { motion } from 'framer-motion'
import { useWebSocketStore } from '../stores/websocketStore'

const TradingChart = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  
  const [timeframe, setTimeframe] = useState('1H')
  const [symbol, setSymbol] = useState('BTCUSDT')
  
  const { marketData, trades } = useWebSocketStore()

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#ffffff',
        fontSize: 12,
        fontFamily: 'Orbitron, monospace',
      },
      grid: {
        vertLines: { color: 'rgba(0, 255, 255, 0.1)' },
        horzLines: { color: 'rgba(0, 255, 255, 0.1)' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#00ffff',
          width: 1,
          style: 2,
        },
        horzLine: {
          color: '#00ffff',
          width: 1,
          style: 2,
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(0, 255, 255, 0.3)',
        textColor: '#ffffff',
      },
      timeScale: {
        borderColor: 'rgba(0, 255, 255, 0.3)',
        textColor: '#ffffff',
        timeVisible: true,
        secondsVisible: false,
      },
      watermark: {
        visible: true,
        fontSize: 48,
        horzAlign: 'center',
        vertAlign: 'center',
        color: 'rgba(0, 255, 255, 0.1)',
        text: 'BACKTESTER',
      },
    })

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff4444',
      borderUpColor: '#00ff88',
      borderDownColor: '#ff4444',
      wickUpColor: '#00ff88',
      wickDownColor: '#ff4444',
    })

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    })

    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries
    volumeSeriesRef.current = volumeSeries

    // Add sample data
    const sampleData = generateSampleData()
    candlestickSeries.setData(sampleData.candlesticks)
    volumeSeries.setData(sampleData.volume)

    // Fit content
    chart.timeScale().fitContent()

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  // Update chart with real market data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !marketData[symbol]) return

    const data = marketData[symbol]
    const time = Math.floor(new Date(data.timestamp).getTime() / 1000)

    // In a real implementation, you would accumulate OHLCV data
    // For demo purposes, we'll simulate it
    candlestickSeriesRef.current.update({
      time,
      open: data.price * 0.999,
      high: data.price * 1.001,
      low: data.price * 0.998,
      close: data.price,
    })
  }, [marketData, symbol])

  // Add trade markers
  useEffect(() => {
    if (!candlestickSeriesRef.current) return

    const recentTrades = trades.slice(0, 50).filter(trade => trade.symbol === symbol)
    
    const markers = recentTrades.map(trade => ({
      time: Math.floor(new Date(trade.timestamp).getTime() / 1000),
      position: trade.side === 'buy' ? 'belowBar' as const : 'aboveBar' as const,
      color: trade.side === 'buy' ? '#00ff88' : '#ff4444',
      shape: trade.side === 'buy' ? 'arrowUp' as const : 'arrowDown' as const,
      text: `${trade.side.toUpperCase()} ${trade.quantity} @ $${trade.price.toFixed(2)}`,
    }))

    candlestickSeriesRef.current.setMarkers(markers)
  }, [trades, symbol])

  const generateSampleData = () => {
    const data = []
    const volumeData = []
    let currentPrice = 50000 + Math.random() * 20000
    const startTime = Math.floor(Date.now() / 1000) - 24 * 60 * 60 // 24 hours ago

    for (let i = 0; i < 100; i++) {
      const time = startTime + i * 60 * 60 // Hourly data
      const open = currentPrice
      const volatility = 0.02
      const change = (Math.random() - 0.5) * volatility
      const high = open * (1 + Math.abs(change) + Math.random() * 0.01)
      const low = open * (1 - Math.abs(change) - Math.random() * 0.01)
      const close = open * (1 + change)
      
      data.push({
        time,
        open,
        high,
        low,
        close,
      })

      volumeData.push({
        time,
        value: Math.random() * 1000000 + 500000,
        color: close > open ? '#00ff8880' : '#ff444480',
      })

      currentPrice = close
    }

    return { candlesticks: data, volume: volumeData }
  }

  const handleTimeframeChange = (_: any, newTimeframe: string | null) => {
    if (newTimeframe) {
      setTimeframe(newTimeframe)
      // In a real implementation, you would fetch new data for the timeframe
    }
  }

  const handleSymbolChange = (_: any, newSymbol: string | null) => {
    if (newSymbol) {
      setSymbol(newSymbol)
      // In a real implementation, you would fetch new data for the symbol
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Card
        sx={{
          height: 600,
          background: 'linear-gradient(145deg, rgba(26, 31, 46, 0.9) 0%, rgba(15, 20, 25, 0.9) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 255, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 255, 255, 0.1)',
        }}
      >
        <CardContent sx={{ height: '100%', p: 2 }}>
          {/* Chart Header */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography
              variant="h6"
              sx={{
                color: 'primary.main',
                fontFamily: 'Orbitron, monospace',
                fontWeight: 'bold',
                textShadow: '0 0 10px rgba(0, 255, 255, 0.3)',
              }}
            >
              {symbol} Price Chart
            </Typography>

            <Box sx={{ display: 'flex', gap: 2 }}>
              {/* Symbol Selector */}
              <ToggleButtonGroup
                value={symbol}
                exclusive
                onChange={handleSymbolChange}
                size="small"
                sx={{
                  '& .MuiToggleButton-root': {
                    color: 'text.secondary',
                    borderColor: 'rgba(0, 255, 255, 0.3)',
                    fontFamily: 'Orbitron, monospace',
                    fontSize: '0.75rem',
                    '&.Mui-selected': {
                      background: 'linear-gradient(45deg, #00ffff, #008888)',
                      color: '#000',
                    },
                  },
                }}
              >
                <ToggleButton value="BTCUSDT">BTC</ToggleButton>
                <ToggleButton value="ETHUSDT">ETH</ToggleButton>
                <ToggleButton value="ADAUSDT">ADA</ToggleButton>
              </ToggleButtonGroup>

              {/* Timeframe Selector */}
              <ToggleButtonGroup
                value={timeframe}
                exclusive
                onChange={handleTimeframeChange}
                size="small"
                sx={{
                  '& .MuiToggleButton-root': {
                    color: 'text.secondary',
                    borderColor: 'rgba(0, 255, 255, 0.3)',
                    fontFamily: 'Orbitron, monospace',
                    fontSize: '0.75rem',
                    '&.Mui-selected': {
                      background: 'linear-gradient(45deg, #00ffff, #008888)',
                      color: '#000',
                    },
                  },
                }}
              >
                <ToggleButton value="1M">1M</ToggleButton>
                <ToggleButton value="5M">5M</ToggleButton>
                <ToggleButton value="15M">15M</ToggleButton>
                <ToggleButton value="1H">1H</ToggleButton>
                <ToggleButton value="4H">4H</ToggleButton>
                <ToggleButton value="1D">1D</ToggleButton>
              </ToggleButtonGroup>
            </Box>
          </Box>

          {/* Chart Container */}
          <Box
            ref={chartContainerRef}
            sx={{
              height: 'calc(100% - 80px)',
              border: '1px solid rgba(0, 255, 255, 0.1)',
              borderRadius: 1,
              overflow: 'hidden',
            }}
          />
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default TradingChart
