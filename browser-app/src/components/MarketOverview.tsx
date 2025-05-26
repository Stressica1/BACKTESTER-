import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Avatar,
  LinearProgress,
} from '@mui/material'
import { TrendingUp, TrendingDown, Timeline } from '@mui/icons-material'
import { motion } from 'framer-motion'
import { useWebSocketStore } from '../stores/websocketStore'

const MarketOverview = () => {
  const { marketData } = useWebSocketStore()

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  // Sample market data if none available
  const sampleMarkets = [
    { symbol: 'BTCUSDT', price: 67245.32, changePercent: 2.45, volume: 28945632 },
    { symbol: 'ETHUSDT', price: 3842.18, changePercent: -1.23, volume: 15234567 },
    { symbol: 'ADAUSDT', price: 0.4521, changePercent: 5.67, volume: 8765432 },
    { symbol: 'DOTUSDT', price: 6.789, changePercent: -0.87, volume: 4523876 },
  ]

  const markets = Object.keys(marketData).length > 0 
    ? Object.values(marketData).slice(0, 4)
    : sampleMarkets

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
                background: 'linear-gradient(45deg, #00ffff, #008888)',
                mr: 2,
              }}
            >
              <Timeline />
            </Avatar>
            <Typography
              variant="h6"
              sx={{
                color: 'primary.main',
                fontFamily: 'Orbitron, monospace',
                fontWeight: 'bold',
                textShadow: '0 0 10px rgba(0, 255, 255, 0.3)',
              }}
            >
              Market Overview
            </Typography>
          </Box>

          <Grid container spacing={2}>
            {markets.map((market, index) => (
              <Grid item xs={12} key={market.symbol}>
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Box
                    sx={{
                      p: 2,
                      border: '1px solid rgba(0, 255, 255, 0.1)',
                      borderRadius: 1,
                      background: 'rgba(26, 31, 46, 0.3)',
                      '&:hover': {
                        background: 'rgba(0, 255, 255, 0.05)',
                        transform: 'translateY(-2px)',
                        transition: 'all 0.3s ease',
                      },
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography
                        variant="subtitle1"
                        sx={{
                          fontFamily: 'Orbitron, monospace',
                          fontWeight: 'bold',
                          color: 'text.primary',
                        }}
                      >
                        {market.symbol.replace('USDT', '/USDT')}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        {market.changePercent >= 0 ? (
                          <TrendingUp sx={{ color: '#00ff88', fontSize: 16 }} />
                        ) : (
                          <TrendingDown sx={{ color: '#ff4444', fontSize: 16 }} />
                        )}
                        <Typography
                          variant="body2"
                          sx={{
                            color: market.changePercent >= 0 ? '#00ff88' : '#ff4444',
                            fontFamily: 'monospace',
                            fontWeight: 'bold',
                          }}
                        >
                          {formatPercent(market.changePercent)}
                        </Typography>
                      </Box>
                    </Box>

                    <Typography
                      variant="h6"
                      sx={{
                        fontFamily: 'monospace',
                        fontWeight: 'bold',
                        color: 'text.primary',
                        mb: 1,
                      }}
                    >
                      {formatCurrency(market.price)}
                    </Typography>

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography
                        variant="caption"
                        sx={{
                          color: 'text.secondary',
                          fontFamily: 'monospace',
                        }}
                      >
                        Vol: {(market.volume / 1000000).toFixed(1)}M
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={Math.abs(market.changePercent) * 10}
                        sx={{
                          width: 60,
                          height: 4,
                          borderRadius: 2,
                          '& .MuiLinearProgress-bar': {
                            background: market.changePercent >= 0 
                              ? 'linear-gradient(45deg, #00ff88, #00cc66)'
                              : 'linear-gradient(45deg, #ff4444, #cc2222)',
                          },
                        }}
                      />
                    </Box>
                  </Box>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default MarketOverview
