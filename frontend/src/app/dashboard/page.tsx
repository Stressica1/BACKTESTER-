'use client';

import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowUpIcon,
  ArrowDownIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  DollarSignIcon,
  BarChart3Icon,
  PieChartIcon,
  ActivityIcon,
  BellIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Loading } from '@/components/ui/Loading';
import { api } from '@/lib/api';
import { useMarketDataStore, useTradingStore } from '@/lib/store';
import { formatCurrency, formatPercentage, formatNumber } from '@/lib/utils';
import { AreaChart, BarChart, LineChart, Heatmap } from '@/components/charts';

interface DashboardMetrics {
  totalBalance: number;
  availableBalance: number;
  totalPnL: number;
  dailyPnL: number;
  totalPositions: number;
  activeOrders: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  portfolioValue: number;
  marginUsed: number;
  marginAvailable: number;
}

interface RecentTrade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  pnl: number;
  timestamp: string;
  status: 'filled' | 'partial' | 'cancelled';
}

interface TopPosition {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercentage: number;
  value: number;
}

interface MarketMover {
  symbol: string;
  price: number;
  change: number;
  changePercentage: number;
  volume: number;
}

interface SystemAlert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  isRead: boolean;
}

export default function DashboardPage() {
  const { marketData } = useMarketDataStore();
  const { positions, orders } = useTradingStore();
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | '3M' | '1Y'>('1D');

  // Fetch dashboard metrics
  const { data: metrics, isLoading: metricsLoading } = useQuery<DashboardMetrics>({
    queryKey: ['dashboard-metrics'],
    queryFn: () => api.get('/trading/metrics'),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch recent trades
  const { data: recentTrades = [], isLoading: tradesLoading } = useQuery<RecentTrade[]>({
    queryKey: ['recent-trades'],
    queryFn: () => api.get('/trading/trades/recent?limit=5'),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch top positions
  const { data: topPositions = [], isLoading: positionsLoading } = useQuery<TopPosition[]>({
    queryKey: ['top-positions'],
    queryFn: () => api.get('/trading/positions/top?limit=5'),
    refetchInterval: 30000,
  });

  // Fetch market movers
  const { data: marketMovers = [], isLoading: moversLoading } = useQuery<MarketMover[]>({
    queryKey: ['market-movers'],
    queryFn: () => api.get('/data/market/movers'),
    refetchInterval: 60000, // Refresh every minute
  });

  // Fetch system alerts
  const { data: systemAlerts = [], isLoading: alertsLoading } = useQuery<SystemAlert[]>({
    queryKey: ['system-alerts'],
    queryFn: () => api.get('/notifications/alerts?limit=5'),
    refetchInterval: 30000,
  });

  // Fetch portfolio performance chart data
  const { data: performanceData, isLoading: performanceLoading } = useQuery({
    queryKey: ['portfolio-performance', timeframe],
    queryFn: () => api.get(`/analytics/portfolio/performance?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  if (metricsLoading) {
    return (
      <div className="p-6">
        <Loading size="lg" className="mx-auto" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header Section */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Trading Dashboard
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Monitor your portfolio performance and market activity
          </p>
        </div>
        <div className="mt-4 lg:mt-0 flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
            {(['1D', '1W', '1M', '3M', '1Y'] as const).map((period) => (
              <button
                key={period}
                onClick={() => setTimeframe(period)}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  timeframe === period
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                {period}
              </button>
            ))}
          </div>
          <Button variant="primary" size="sm">
            <ActivityIcon className="w-4 h-4 mr-2" />
            Live Trading
          </Button>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Portfolio Value
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(metrics?.portfolioValue || 0)}
              </p>
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-full">
              <DollarSignIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
          <div className="mt-4 flex items-center">
            {(metrics?.dailyPnL || 0) >= 0 ? (
              <ArrowUpIcon className="w-4 h-4 text-green-500 mr-1" />
            ) : (
              <ArrowDownIcon className="w-4 h-4 text-red-500 mr-1" />
            )}
            <span
              className={`text-sm font-medium ${
                (metrics?.dailyPnL || 0) >= 0 ? 'text-green-600' : 'text-red-600'
              }`}
            >
              {formatCurrency(metrics?.dailyPnL || 0)} today
            </span>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total P&L
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(metrics?.totalPnL || 0)}
              </p>
            </div>
            <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-full">
              <TrendingUpIcon className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Win Rate: <span className="font-medium">{formatPercentage(metrics?.winRate || 0)}</span>
            </span>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Active Positions
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {metrics?.totalPositions || 0}
              </p>
            </div>
            <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-full">
              <BarChart3Icon className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Orders: <span className="font-medium">{metrics?.activeOrders || 0}</span>
            </span>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Available Balance
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(metrics?.availableBalance || 0)}
              </p>
            </div>
            <div className="p-3 bg-orange-100 dark:bg-orange-900/20 rounded-full">
              <PieChartIcon className="w-6 h-6 text-orange-600 dark:text-orange-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Margin Used: <span className="font-medium">{formatPercentage((metrics?.marginUsed || 0) / (metrics?.portfolioValue || 1))}</span>
            </span>
          </div>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Portfolio Performance Chart */}
        <Card className="lg:col-span-2 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Portfolio Performance
            </h3>
            <div className="flex items-center space-x-2 text-sm">
              <span className="text-gray-600 dark:text-gray-400">Sharpe Ratio:</span>
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {metrics?.sharpeRatio?.toFixed(2) || 'N/A'}
              </span>
            </div>
          </div>          {performanceLoading ? (
            <div className="h-64 flex items-center justify-center">
              <Loading size="md" />
            </div>
          ) : (
            <AreaChart
              data={{
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [
                  {
                    label: 'Portfolio Value',
                    data: [100000, 105000, 98000, 112000, 118000, metrics?.portfolioValue || 120000],
                    borderColor: '#3B82F6',
                    backgroundColor: '#3B82F640',
                    fill: true,
                  },
                ],
              }}
              height="256px"
              showLegend={false}
              gradientFill={true}
            />
          )}
        </Card>

        {/* System Alerts */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              System Alerts
            </h3>
            <BellIcon className="w-5 h-5 text-gray-400" />
          </div>
          {alertsLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-4">
              {systemAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                >
                  <div className="flex-shrink-0">
                    {alert.type === 'success' && (
                      <CheckCircleIcon className="w-5 h-5 text-green-500" />
                    )}
                    {alert.type === 'warning' && (
                      <AlertTriangleIcon className="w-5 h-5 text-yellow-500" />
                    )}
                    {alert.type === 'error' && (
                      <XCircleIcon className="w-5 h-5 text-red-500" />
                    )}
                    {alert.type === 'info' && (
                      <BellIcon className="w-5 h-5 text-blue-500" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {alert.title}
                    </p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                      {alert.message}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {systemAlerts.length === 0 && (
                <div className="text-center py-8">
                  <BellIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-600 dark:text-gray-400">No alerts</p>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      {/* Recent Activity Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Trades */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Recent Trades
            </h3>
            <Button variant="outline" size="sm">
              View All
            </Button>
          </div>
          {tradesLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-4">
              {recentTrades.map((trade) => (
                <div
                  key={trade.id}
                  className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        trade.side === 'buy' ? 'bg-green-500' : 'bg-red-500'
                      }`}
                    />
                    <div>
                      <p className="font-medium text-gray-900 dark:text-gray-100">
                        {trade.symbol}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {trade.side.toUpperCase()} {formatNumber(trade.quantity)} @ {formatCurrency(trade.price)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p
                      className={`font-medium ${
                        trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {formatCurrency(trade.pnl)}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-500">
                      {new Date(trade.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {recentTrades.length === 0 && (
                <div className="text-center py-8">
                  <ActivityIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-600 dark:text-gray-400">No recent trades</p>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Top Positions */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Top Positions
            </h3>
            <Button variant="outline" size="sm">
              View All
            </Button>
          </div>
          {positionsLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-4">
              {topPositions.map((position) => (
                <div
                  key={position.symbol}
                  className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
                >
                  <div>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {position.symbol}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {formatNumber(position.quantity)} shares @ {formatCurrency(position.avgPrice)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p
                      className={`font-medium ${
                        position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {formatCurrency(position.pnl)}
                    </p>
                    <p
                      className={`text-sm ${
                        position.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {formatPercentage(position.pnlPercentage)}
                    </p>
                  </div>
                </div>
              ))}
              {topPositions.length === 0 && (
                <div className="text-center py-8">
                  <PieChartIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-600 dark:text-gray-400">No positions</p>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      {/* Market Movers */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Market Movers
          </h3>
          <Button variant="outline" size="sm">
            View Market
          </Button>
        </div>
        {moversLoading ? (
          <Loading size="sm" />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {marketMovers.map((mover) => (
              <div
                key={mover.symbol}
                className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors cursor-pointer"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {mover.symbol}
                  </span>
                  {mover.changePercentage >= 0 ? (
                    <TrendingUpIcon className="w-4 h-4 text-green-500" />
                  ) : (
                    <TrendingDownIcon className="w-4 h-4 text-red-500" />
                  )}
                </div>
                <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                  {formatCurrency(mover.price)}
                </p>
                <p
                  className={`text-sm ${
                    mover.changePercentage >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {formatCurrency(mover.change)} ({formatPercentage(mover.changePercentage)})
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                  Vol: {formatNumber(mover.volume, { compact: true })}
                </p>
              </div>
            ))}
            {marketMovers.length === 0 && (
              <div className="col-span-full text-center py-8">
                <BarChart3Icon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-600 dark:text-gray-400">No market data available</p>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
