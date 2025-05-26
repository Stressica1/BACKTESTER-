import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Avatar,
  Chip,
} from '@mui/material'
import {
  AccountBalance,
  TrendingUp,
  TrendingDown,
  ShowChart,
  Assessment,
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import { useWebSocketStore } from '../stores/websocketStore'

const PerformanceMetrics = () => {
  const { backtestResults, trades } = useWebSocketStore()

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  const formatPercent = (value: number) => {
    return `${value.toFixed(2)}%`
  }

  // Calculate metrics from trades if no backtest results
  const calculateMetrics = () => {
    if (backtestResults) {
      return backtestResults
    }

    const closedTrades = trades.filter(trade => trade.status === 'closed' && trade.profit !== undefined)
    const totalTrades = closedTrades.length
    const winningTrades = closedTrades.filter(trade => (trade.profit || 0) > 0)
    const totalPnL = closedTrades.reduce((sum, trade) => sum + (trade.profit || 0), 0)
    
    return {
      totalTrades,
      winRate: totalTrades > 0 ? (winningTrades.length / totalTrades) * 100 : 0,
      totalPnL,
      maxDrawdown: -15.5, // Sample data
      sharpeRatio: 1.85, // Sample data
      startDate: '2024-01-01',
      endDate: new Date().toISOString().split('T')[0],
    }
  }

  const metrics = calculateMetrics()

  const metricCards = [
    {
      title: 'Total P&L',
      value: formatCurrency(metrics.totalPnL),
      icon: AccountBalance,
      color: metrics.totalPnL >= 0 ? '#00ff88' : '#ff4444',
      subtitle: 'Overall Performance',
    },
    {
      title: 'Win Rate',
      value: formatPercent(metrics.winRate),
      icon: TrendingUp,
      color: metrics.winRate >= 50 ? '#00ff88' : '#ff4444',
      subtitle: `${metrics.totalTrades} Total Trades`,
    },
    {
      title: 'Max Drawdown',
      value: formatPercent(metrics.maxDrawdown),
      icon: TrendingDown,
      color: '#ff4444',
      subtitle: 'Risk Metric',
    },
    {
      title: 'Sharpe Ratio',
      value: metrics.sharpeRatio.toFixed(2),
      icon: ShowChart,
      color: metrics.sharpeRatio >= 1 ? '#00ff88' : '#ffaa00',
      subtitle: 'Risk-Adjusted Return',
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card
        sx={{
          background: 'linear-gradient(145deg, rgba(26, 31, 46, 0.9) 0%, rgba(15, 20, 25, 0.9) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 255, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 255, 255, 0.1)',
          height: '100%',
        }}
      >
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Avatar
              sx={{
                background: 'linear-gradient(45deg, #ff00ff, #880088)',
                mr: 2,
              }}
            >
              <Assessment />
            </Avatar>
            <Typography
              variant="h6"
              sx={{
                color: 'secondary.main',
                fontFamily: 'Orbitron, monospace',
                fontWeight: 'bold',
                textShadow: '0 0 10px rgba(255, 0, 255, 0.3)',
              }}
            >
              Performance Metrics
            </Typography>
          </Box>

          <Grid container spacing={2}>
            {metricCards.map((metric, index) => (
              <Grid item xs={12} sm={6} key={metric.title}>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Box
                    sx={{
                      p: 2,
                      border: '1px solid rgba(0, 255, 255, 0.1)',
                      borderRadius: 1,
                      background: 'rgba(26, 31, 46, 0.3)',
                      textAlign: 'center',
                      '&:hover': {
                        background: 'rgba(0, 255, 255, 0.05)',
                        transform: 'scale(1.02)',
                        transition: 'all 0.3s ease',
                      },
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          background: `linear-gradient(45deg, ${metric.color}, ${metric.color}88)`,
                        }}
                      >
                        <metric.icon sx={{ fontSize: 18 }} />
                      </Avatar>
                    </Box>

                    <Typography
                      variant="caption"
                      sx={{
                        color: 'text.secondary',
                        fontFamily: 'Orbitron, monospace',
                        textTransform: 'uppercase',
                        letterSpacing: 1,
                      }}
                    >
                      {metric.title}
                    </Typography>

                    <Typography
                      variant="h5"
                      sx={{
                        color: metric.color,
                        fontFamily: 'monospace',
                        fontWeight: 'bold',
                        my: 1,
                        textShadow: `0 0 10px ${metric.color}40`,
                      }}
                    >
                      {metric.value}
                    </Typography>

                    <Typography
                      variant="caption"
                      sx={{
                        color: 'text.secondary',
                        fontSize: '0.7rem',
                      }}
                    >
                      {metric.subtitle}
                    </Typography>
                  </Box>
                </motion.div>
              </Grid>
            ))}
          </Grid>

          {/* Performance Summary */}
          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Typography
              variant="caption"
              sx={{
                color: 'text.secondary',
                fontFamily: 'monospace',
                display: 'block',
                mb: 1,
              }}
            >
              Backtest Period: {metrics.startDate} to {metrics.endDate}
            </Typography>
            
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap' }}>
              <Chip
                label={`${trades.filter(t => t.status === 'open').length} Open Positions`}
                size="small"
                color="warning"
                variant="outlined"
                sx={{ fontFamily: 'Orbitron, monospace', fontSize: '0.7rem' }}
              />
              <Chip
                label={`${trades.filter(t => t.status === 'closed').length} Closed Trades`}
                size="small"
                color="info"
                variant="outlined"
                sx={{ fontFamily: 'Orbitron, monospace', fontSize: '0.7rem' }}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default PerformanceMetrics
