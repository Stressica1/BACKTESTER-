import {
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  LinearProgress,
  Chip,
  Grid,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Settings,
  TrendingUp,
  Speed,
  Timeline,
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import { useState } from 'react'
import { useWebSocketStore } from '../stores/websocketStore'

const BacktestPanel = () => {
  const { sendMessage, connectionStatus, addAlert } = useWebSocketStore()
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  
  const [config, setConfig] = useState({
    symbol: 'BTCUSDT',
    strategy: 'mean_reversion',
    timeframe: '1h',
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    initialCapital: 10000,
    stopLoss: 2,
    takeProfit: 4,
    maxPositions: 3,
  })

  const strategies = [
    { value: 'mean_reversion', label: 'Mean Reversion' },
    { value: 'momentum', label: 'Momentum' },
    { value: 'breakout', label: 'Breakout' },
    { value: 'scalping', label: 'Scalping' },
    { value: 'swing_trading', label: 'Swing Trading' },
  ]

  const symbols = [
    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 
    'LINKUSDT', 'LTCUSDT', 'XRPUSDT', 'BNBUSDT'
  ]

  const timeframes = [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
  ]

  const handleStartBacktest = () => {
    if (connectionStatus.status !== 'connected') {
      addAlert('Cannot start backtest: WebSocket not connected', 'error')
      return
    }

    setIsRunning(true)
    setProgress(0)
    
    const message = {
      type: 'start_backtest',
      config: {
        ...config,
        initial_capital: config.initialCapital,
        stop_loss: config.stopLoss / 100,
        take_profit: config.takeProfit / 100,
        max_positions: config.maxPositions,
      },
    }

    sendMessage(message)
    addAlert('Backtest started successfully', 'success')

    // Simulate progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsRunning(false)
          return 100
        }
        return prev + Math.random() * 10
      })
    }, 1000)
  }

  const handleStopBacktest = () => {
    setIsRunning(false)
    setProgress(0)
    
    sendMessage({ type: 'stop_backtest' })
    addAlert('Backtest stopped', 'warning')
  }

  const handleConfigChange = (field: string, value: string | number) => {
    setConfig(prev => ({
      ...prev,
      [field]: value,
    }))
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card
        sx={{
          background: 'linear-gradient(145deg, rgba(26, 31, 46, 0.9) 0%, rgba(15, 20, 25, 0.9) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 255, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 255, 255, 0.1)',
          height: 'fit-content',
          position: 'sticky',
          top: 20,
        }}
      >
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Settings sx={{ color: 'primary.main', mr: 1 }} />
            <Typography
              variant="h6"
              sx={{
                color: 'primary.main',
                fontFamily: 'Orbitron, monospace',
                fontWeight: 'bold',
                textShadow: '0 0 10px rgba(0, 255, 255, 0.3)',
              }}
            >
              Backtest Configuration
            </Typography>
          </Box>

          {/* Status */}
          {isRunning && (
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Running Backtest...
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {Math.round(progress)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  '& .MuiLinearProgress-bar': {
                    background: 'linear-gradient(45deg, #00ffff, #ff00ff)',
                  },
                }}
              />
            </Box>
          )}

          {/* Strategy Selection */}
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Strategy</InputLabel>
            <Select
              value={config.strategy}
              label="Strategy"
              onChange={(e) => handleConfigChange('strategy', e.target.value)}
              sx={{
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 255, 255, 0.3)',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(0, 255, 255, 0.5)',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'primary.main',
                },
              }}
            >
              {strategies.map((strategy) => (
                <MenuItem key={strategy.value} value={strategy.value}>
                  {strategy.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Symbol and Timeframe */}
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Symbol</InputLabel>
                <Select
                  value={config.symbol}
                  label="Symbol"
                  onChange={(e) => handleConfigChange('symbol', e.target.value)}
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 255, 255, 0.3)',
                    },
                  }}
                >
                  {symbols.map((symbol) => (
                    <MenuItem key={symbol} value={symbol}>
                      {symbol}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Timeframe</InputLabel>
                <Select
                  value={config.timeframe}
                  label="Timeframe"
                  onChange={(e) => handleConfigChange('timeframe', e.target.value)}
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 255, 255, 0.3)',
                    },
                  }}
                >
                  {timeframes.map((tf) => (
                    <MenuItem key={tf.value} value={tf.value}>
                      {tf.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>

          {/* Date Range */}
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Start Date"
                type="date"
                value={config.startDate}
                onChange={(e) => handleConfigChange('startDate', e.target.value)}
                InputLabelProps={{ shrink: true }}
                sx={{
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(0, 255, 255, 0.3)',
                  },
                }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="End Date"
                type="date"
                value={config.endDate}
                onChange={(e) => handleConfigChange('endDate', e.target.value)}
                InputLabelProps={{ shrink: true }}
                sx={{
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(0, 255, 255, 0.3)',
                  },
                }}
              />
            </Grid>
          </Grid>

          {/* Capital and Risk Management */}
          <TextField
            fullWidth
            label="Initial Capital ($)"
            type="number"
            value={config.initialCapital}
            onChange={(e) => handleConfigChange('initialCapital', parseFloat(e.target.value))}
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(0, 255, 255, 0.3)',
              },
            }}
          />

          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={4}>
              <TextField
                fullWidth
                label="Stop Loss (%)"
                type="number"
                value={config.stopLoss}
                onChange={(e) => handleConfigChange('stopLoss', parseFloat(e.target.value))}
                sx={{
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(255, 68, 68, 0.3)',
                  },
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                label="Take Profit (%)"
                type="number"
                value={config.takeProfit}
                onChange={(e) => handleConfigChange('takeProfit', parseFloat(e.target.value))}
                sx={{
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(0, 255, 136, 0.3)',
                  },
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                label="Max Positions"
                type="number"
                value={config.maxPositions}
                onChange={(e) => handleConfigChange('maxPositions', parseInt(e.target.value))}
                sx={{
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(0, 255, 255, 0.3)',
                  },
                }}
              />
            </Grid>
          </Grid>

          <Divider sx={{ my: 2, borderColor: 'rgba(0, 255, 255, 0.2)' }} />

          {/* Control Buttons */}
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            {!isRunning ? (
              <Button
                variant="contained"
                startIcon={<PlayArrow />}
                onClick={handleStartBacktest}
                disabled={connectionStatus.status !== 'connected'}
                fullWidth
                sx={{
                  background: 'linear-gradient(45deg, #00ff88, #00cc66)',
                  color: '#000',
                  fontFamily: 'Orbitron, monospace',
                  fontWeight: 'bold',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #00cc66, #009944)',
                  },
                  '&:disabled': {
                    background: 'rgba(128, 128, 128, 0.3)',
                    color: 'rgba(255, 255, 255, 0.3)',
                  },
                }}
              >
                Start Backtest
              </Button>
            ) : (
              <Button
                variant="contained"
                startIcon={<Stop />}
                onClick={handleStopBacktest}
                fullWidth
                sx={{
                  background: 'linear-gradient(45deg, #ff4444, #cc2222)',
                  color: '#fff',
                  fontFamily: 'Orbitron, monospace',
                  fontWeight: 'bold',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #cc2222, #991111)',
                  },
                }}
              >
                Stop Backtest
              </Button>
            )}
          </Box>

          {/* Quick Stats */}
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              icon={<TrendingUp />}
              label={`$${config.initialCapital.toLocaleString()}`}
              size="small"
              variant="outlined"
              sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}
            />
            <Chip
              icon={<Speed />}
              label={`${config.timeframe.toUpperCase()}`}
              size="small"
              variant="outlined"
              sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}
            />
            <Chip
              icon={<Timeline />}
              label={config.strategy.replace('_', ' ').toUpperCase()}
              size="small"
              variant="outlined"
              sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}
            />
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default BacktestPanel
