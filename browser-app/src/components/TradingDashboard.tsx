import { Grid, Container, Box, Typography, Fade } from '@mui/material'
import { motion } from 'framer-motion'
import ConnectionStatus from './ConnectionStatus'
import TradingChart from './TradingChart'
import TradesList from './TradesList'
import MarketOverview from './MarketOverview'
import BacktestPanel from './BacktestPanel'
import AlertsPanel from './AlertsPanel'
import PerformanceMetrics from './PerformanceMetrics'

const TradingDashboard = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
      },
    },
  }

  return (
    <Container maxWidth={false} sx={{ py: 2 }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header */}
        <motion.div variants={itemVariants}>
          <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography
              variant="h3"
              component="h1"
              sx={{
                fontWeight: 900,
                background: 'linear-gradient(45deg, #00ffff, #ff00ff)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: '0 0 20px rgba(0, 255, 255, 0.3)',
              }}
            >
              BACKTESTER Pro
            </Typography>
            <ConnectionStatus />
          </Box>
        </motion.div>

        {/* Alerts Panel */}
        <motion.div variants={itemVariants}>
          <AlertsPanel />
        </motion.div>

        {/* Main Dashboard Grid */}
        <Grid container spacing={3}>
          {/* Left Column */}
          <Grid item xs={12} lg={8}>
            <Grid container spacing={3}>
              {/* Trading Chart */}
              <Grid item xs={12}>
                <motion.div variants={itemVariants}>
                  <TradingChart />
                </motion.div>
              </Grid>

              {/* Market Overview */}
              <Grid item xs={12} md={6}>
                <motion.div variants={itemVariants}>
                  <MarketOverview />
                </motion.div>
              </Grid>

              {/* Performance Metrics */}
              <Grid item xs={12} md={6}>
                <motion.div variants={itemVariants}>
                  <PerformanceMetrics />
                </motion.div>
              </Grid>

              {/* Trades List */}
              <Grid item xs={12}>
                <motion.div variants={itemVariants}>
                  <TradesList />
                </motion.div>
              </Grid>
            </Grid>
          </Grid>

          {/* Right Column */}
          <Grid item xs={12} lg={4}>
            <motion.div variants={itemVariants}>
              <BacktestPanel />
            </motion.div>
          </Grid>
        </Grid>
      </motion.div>
    </Container>
  )
}

export default TradingDashboard
