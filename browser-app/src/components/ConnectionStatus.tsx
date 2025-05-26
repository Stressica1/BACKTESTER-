import { Box, Chip, Tooltip, IconButton } from '@mui/material'
import { motion } from 'framer-motion'
import {
  WifiOff,
  Wifi,
  Sync,
  Error as ErrorIcon,
  Refresh,
} from '@mui/icons-material'
import { useWebSocketStore } from '../stores/websocketStore'

const ConnectionStatus = () => {
  const { connectionStatus, connect, disconnect } = useWebSocketStore()

  const getStatusConfig = () => {
    switch (connectionStatus.status) {
      case 'connected':
        return {
          label: 'Connected',
          color: 'success' as const,
          icon: <Wifi />,
          pulse: false,
        }
      case 'connecting':
        return {
          label: 'Connecting...',
          color: 'warning' as const,
          icon: <Sync className="animate-spin" />,
          pulse: true,
        }
      case 'disconnected':
        return {
          label: 'Disconnected',
          color: 'default' as const,
          icon: <WifiOff />,
          pulse: false,
        }
      case 'error':
        return {
          label: 'Connection Error',
          color: 'error' as const,
          icon: <ErrorIcon />,
          pulse: true,
        }
      default:
        return {
          label: 'Unknown',
          color: 'default' as const,
          icon: <WifiOff />,
          pulse: false,
        }
    }
  }

  const config = getStatusConfig()

  const handleReconnect = () => {
    if (connectionStatus.status !== 'connecting') {
      connect('ws://localhost:5000/ws')
    }
  }

  const handleDisconnect = () => {
    disconnect()
  }

  const formatLastHeartbeat = () => {
    if (!connectionStatus.lastHeartbeat) return 'Never'
    const diff = Date.now() - new Date(connectionStatus.lastHeartbeat).getTime()
    if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    return `${Math.floor(diff / 3600000)}h ago`
  }

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Tooltip
        title={
          <Box>
            <div>Status: {config.label}</div>
            <div>Last Heartbeat: {formatLastHeartbeat()}</div>
            <div>Reconnect Attempts: {connectionStatus.reconnectAttempts}</div>
          </Box>
        }
        arrow
      >
        <motion.div
          animate={config.pulse ? { scale: [1, 1.1, 1] } : {}}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          <Chip
            icon={config.icon}
            label={config.label}
            color={config.color}
            variant="filled"
            sx={{
              fontFamily: 'Orbitron, monospace',
              fontWeight: 'bold',
              '& .MuiChip-icon': {
                fontSize: '1.2rem',
              },
              ...(config.color === 'success' && {
                background: 'linear-gradient(45deg, #00ff88, #00ccaa)',
                color: '#001a0f',
                boxShadow: '0 0 10px rgba(0, 255, 136, 0.3)',
              }),
              ...(config.color === 'warning' && {
                background: 'linear-gradient(45deg, #ffaa00, #ff8800)',
                color: '#1a1100',
                boxShadow: '0 0 10px rgba(255, 170, 0, 0.3)',
              }),
              ...(config.color === 'error' && {
                background: 'linear-gradient(45deg, #ff4444, #cc2222)',
                color: '#1a0000',
                boxShadow: '0 0 10px rgba(255, 68, 68, 0.3)',
              }),
            }}
          />
        </motion.div>
      </Tooltip>

      <Box sx={{ display: 'flex', gap: 0.5 }}>
        <Tooltip title="Reconnect">
          <IconButton
            size="small"
            onClick={handleReconnect}
            disabled={connectionStatus.status === 'connecting'}
            sx={{
              color: 'primary.main',
              '&:hover': {
                background: 'rgba(0, 255, 255, 0.1)',
              },
            }}
          >
            <Refresh />
          </IconButton>
        </Tooltip>

        {connectionStatus.status === 'connected' && (
          <Tooltip title="Disconnect">
            <IconButton
              size="small"
              onClick={handleDisconnect}
              sx={{
                color: 'error.main',
                '&:hover': {
                  background: 'rgba(255, 68, 68, 0.1)',
                },
              }}
            >
              <WifiOff />
            </IconButton>
          </Tooltip>
        )}
      </Box>
    </Box>
  )
}

export default ConnectionStatus
