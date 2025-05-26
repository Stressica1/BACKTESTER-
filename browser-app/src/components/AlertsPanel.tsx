import { Alert, AlertTitle, IconButton, Box } from '@mui/material'
import { Close, ClearAll } from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import { useWebSocketStore } from '../stores/websocketStore'

const AlertsPanel = () => {
  const { alerts, removeAlert, clearAlerts } = useWebSocketStore()

  if (alerts.length === 0) return null

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'success': return '✓'
      case 'warning': return '⚠'
      case 'error': return '✗'
      default: return 'ℹ'
    }
  }

  return (
    <Box sx={{ mb: 2 }}>
      <AnimatePresence>
        {alerts.map((alert, index) => (
          <motion.div
            key={alert.id}
            initial={{ opacity: 0, height: 0, marginBottom: 0 }}
            animate={{ opacity: 1, height: 'auto', marginBottom: 8 }}
            exit={{ opacity: 0, height: 0, marginBottom: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Alert
              severity={alert.type}
              action={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {index === 0 && alerts.length > 1 && (
                    <IconButton
                      size="small"
                      onClick={clearAlerts}
                      sx={{ color: 'inherit', mr: 1 }}
                    >
                      <ClearAll fontSize="small" />
                    </IconButton>
                  )}
                  <IconButton
                    size="small"
                    onClick={() => removeAlert(alert.id)}
                    sx={{ color: 'inherit' }}
                  >
                    <Close fontSize="small" />
                  </IconButton>
                </Box>
              }
              sx={{
                fontFamily: 'Orbitron, monospace',
                '& .MuiAlert-icon': {
                  fontSize: '1.2rem',
                },
                '& .MuiAlert-message': {
                  fontSize: '0.9rem',
                },
                background: () => {
                  const colors = {
                    info: 'linear-gradient(45deg, rgba(33, 150, 243, 0.1), rgba(33, 150, 243, 0.05))',
                    success: 'linear-gradient(45deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.05))',
                    warning: 'linear-gradient(45deg, rgba(255, 152, 0, 0.1), rgba(255, 152, 0, 0.05))',
                    error: 'linear-gradient(45deg, rgba(244, 67, 54, 0.1), rgba(244, 67, 54, 0.05))',
                  }
                  return colors[alert.type as keyof typeof colors]
                },
                border: () => {
                  const colors = {
                    info: '1px solid rgba(33, 150, 243, 0.3)',
                    success: '1px solid rgba(76, 175, 80, 0.3)',
                    warning: '1px solid rgba(255, 152, 0, 0.3)',
                    error: '1px solid rgba(244, 67, 54, 0.3)',
                  }
                  return colors[alert.type as keyof typeof colors]
                },
                boxShadow: () => {
                  const colors = {
                    info: '0 4px 20px rgba(33, 150, 243, 0.1)',
                    success: '0 4px 20px rgba(76, 175, 80, 0.1)',
                    warning: '0 4px 20px rgba(255, 152, 0, 0.1)',
                    error: '0 4px 20px rgba(244, 67, 54, 0.1)',
                  }
                  return colors[alert.type as keyof typeof colors]
                },
              }}
            >
              <AlertTitle sx={{ fontWeight: 'bold', fontSize: '1rem' }}>
                {getAlertIcon(alert.type)} {alert.type.toUpperCase()}
              </AlertTitle>
              {alert.message}
              <Box
                component="span"
                sx={{
                  display: 'block',
                  fontSize: '0.75rem',
                  opacity: 0.7,
                  mt: 0.5,
                }}
              >
                {new Date(alert.timestamp).toLocaleTimeString()}
              </Box>
            </Alert>
          </motion.div>
        ))}
      </AnimatePresence>
    </Box>
  )
}

export default AlertsPanel
