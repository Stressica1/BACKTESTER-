import {
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Chip,
  Box,
  Avatar,
} from '@mui/material'
import { TrendingUp, TrendingDown } from '@mui/icons-material'
import { motion } from 'framer-motion'
import { useWebSocketStore } from '../stores/websocketStore'

const TradesList = () => {
  const { trades } = useWebSocketStore()

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const getProfitColor = (profit?: number) => {
    if (!profit) return 'text.secondary'
    return profit > 0 ? '#00ff88' : '#ff4444'
  }

  const recentTrades = trades.slice(0, 20) // Show last 20 trades

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
        }}
      >
        <CardContent>
          <Typography
            variant="h6"
            sx={{
              mb: 2,
              color: 'primary.main',
              fontFamily: 'Orbitron, monospace',
              fontWeight: 'bold',
              textShadow: '0 0 10px rgba(0, 255, 255, 0.3)',
            }}
          >
            Recent Trades ({trades.length})
          </Typography>

          <TableContainer sx={{ maxHeight: 400 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell
                    sx={{
                      background: 'rgba(26, 31, 46, 0.9)',
                      color: 'primary.main',
                      fontFamily: 'Orbitron, monospace',
                      fontWeight: 'bold',
                      borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
                    }}
                  >
                    Time
                  </TableCell>
                  <TableCell
                    sx={{
                      background: 'rgba(26, 31, 46, 0.9)',
                      color: 'primary.main',
                      fontFamily: 'Orbitron, monospace',
                      fontWeight: 'bold',
                      borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
                    }}
                  >
                    Symbol
                  </TableCell>
                  <TableCell
                    sx={{
                      background: 'rgba(26, 31, 46, 0.9)',
                      color: 'primary.main',
                      fontFamily: 'Orbitron, monospace',
                      fontWeight: 'bold',
                      borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
                    }}
                  >
                    Side
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{
                      background: 'rgba(26, 31, 46, 0.9)',
                      color: 'primary.main',
                      fontFamily: 'Orbitron, monospace',
                      fontWeight: 'bold',
                      borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
                    }}
                  >
                    Quantity
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{
                      background: 'rgba(26, 31, 46, 0.9)',
                      color: 'primary.main',
                      fontFamily: 'Orbitron, monospace',
                      fontWeight: 'bold',
                      borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
                    }}
                  >
                    Price
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{
                      background: 'rgba(26, 31, 46, 0.9)',
                      color: 'primary.main',
                      fontFamily: 'Orbitron, monospace',
                      fontWeight: 'bold',
                      borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
                    }}
                  >
                    P&L
                  </TableCell>
                  <TableCell
                    sx={{
                      background: 'rgba(26, 31, 46, 0.9)',
                      color: 'primary.main',
                      fontFamily: 'Orbitron, monospace',
                      fontWeight: 'bold',
                      borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
                    }}
                  >
                    Status
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {recentTrades.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={7}
                      align="center"
                      sx={{
                        py: 4,
                        color: 'text.secondary',
                        fontStyle: 'italic',
                      }}
                    >
                      No trades yet. Start your backtesting to see trades here.
                    </TableCell>
                  </TableRow>
                ) : (
                  recentTrades.map((trade, index) => (
                    <motion.tr
                      key={trade.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      component={TableRow}
                      sx={{
                        '&:hover': {
                          background: 'rgba(0, 255, 255, 0.05)',
                        },
                      }}
                    >
                      <TableCell
                        sx={{
                          color: 'text.secondary',
                          fontSize: '0.8rem',
                          fontFamily: 'monospace',
                        }}
                      >
                        {formatTime(trade.timestamp)}
                      </TableCell>
                      <TableCell
                        sx={{
                          color: 'text.primary',
                          fontFamily: 'Orbitron, monospace',
                          fontWeight: 'bold',
                        }}
                      >
                        {trade.symbol}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Avatar
                            sx={{
                              width: 20,
                              height: 20,
                              background: trade.side === 'buy' ? '#00ff88' : '#ff4444',
                            }}
                          >
                            {trade.side === 'buy' ? (
                              <TrendingUp sx={{ fontSize: 12 }} />
                            ) : (
                              <TrendingDown sx={{ fontSize: 12 }} />
                            )}
                          </Avatar>
                          <Typography
                            variant="body2"
                            sx={{
                              color: trade.side === 'buy' ? '#00ff88' : '#ff4444',
                              fontFamily: 'Orbitron, monospace',
                              fontWeight: 'bold',
                              textTransform: 'uppercase',
                            }}
                          >
                            {trade.side}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          color: 'text.primary',
                          fontFamily: 'monospace',
                        }}
                      >
                        {trade.quantity.toFixed(6)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          color: 'text.primary',
                          fontFamily: 'monospace',
                        }}
                      >
                        {formatCurrency(trade.price)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          color: getProfitColor(trade.profit),
                          fontFamily: 'monospace',
                          fontWeight: 'bold',
                        }}
                      >
                        {trade.profit ? (
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                            {trade.profit > 0 ? (
                              <TrendingUp sx={{ fontSize: 16 }} />
                            ) : (
                              <TrendingDown sx={{ fontSize: 16 }} />
                            )}
                            {formatCurrency(Math.abs(trade.profit))}
                          </Box>
                        ) : (
                          '-'
                        )}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={trade.status}
                          size="small"
                          color={trade.status === 'open' ? 'warning' : 'success'}
                          variant="outlined"
                          sx={{
                            fontFamily: 'Orbitron, monospace',
                            fontSize: '0.7rem',
                            fontWeight: 'bold',
                          }}
                        />
                      </TableCell>
                    </motion.tr>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default TradesList
