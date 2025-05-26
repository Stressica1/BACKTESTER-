import { useEffect } from 'react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { motion } from 'framer-motion'
import TradingDashboard from './components/TradingDashboard'
import { useWebSocketStore } from './stores/websocketStore'
import './App.css'

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ffff',
    },
    secondary: {
      main: '#ff00ff',
    },
    background: {
      default: '#0a0e1a',
      paper: '#1a1f2e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Orbitron", "Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(145deg, #1a1f2e 0%, #0f1419 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 255, 255, 0.1)',
          boxShadow: '0 4px 20px rgba(0, 255, 255, 0.1)',
        },
      },
    },
  },
})

function App() {
  const { connect, disconnect } = useWebSocketStore()

  useEffect(() => {
    console.log('🚀 App Component Mounted - Connecting to WebSocket...')
    // Connect to WebSocket when app starts
    connect('ws://localhost:5000/ws')
    
    // Cleanup on unmount
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="app"
      >
        <TradingDashboard />
      </motion.div>
    </ThemeProvider>
  )
}

export default App
