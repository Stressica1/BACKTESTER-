'use client';

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart3Icon,
  TrendingUpIcon,
  PieChartIcon,
  ActivityIcon,
  CalendarIcon,
  FilterIcon,
  DownloadIcon,
  RefreshCwIcon,
  EyeIcon,
  Settings2Icon,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Loading } from '@/components/ui/Loading';
import { Badge } from '@/components/ui/Badge';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table';
import { api } from '@/lib/api';
import { formatCurrency, formatNumber, formatPercentage } from '@/lib/utils';
import { LineChart, AreaChart, BarChart, Heatmap } from '@/components/charts';

interface AnalyticsOverview {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnL: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  returnsCorrelation: number;
}

interface TradeAnalysis {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercentage: number;
  holdingPeriod: number;
  entryTime: string;
  exitTime: string;
  strategy?: string;
  commissions: number;
}

interface PerformanceMetric {
  name: string;
  value: number;
  benchmark?: number;
  change: number;
  isPercentage: boolean;
  trend: 'up' | 'down' | 'neutral';
}

interface DrawdownPeriod {
  startDate: string;
  endDate: string;
  duration: number;
  maxDrawdown: number;
  recovery: boolean;
}

interface MonthlyReturns {
  month: string;
  returns: number;
  benchmark?: number;
  trades: number;
}

interface RiskMetrics {
  var95: number;
  var99: number;
  expectedShortfall: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  trackingError: number;
  correlation: number;
}

export default function AnalyticsPage() {
  const [timeframe, setTimeframe] = useState<'1M' | '3M' | '6M' | '1Y' | 'ALL'>('3M');
  const [analysisType, setAnalysisType] = useState<'performance' | 'risk' | 'trades' | 'drawdown'>('performance');

  // Fetch analytics overview
  const { data: overview, isLoading: overviewLoading } = useQuery<AnalyticsOverview>({
    queryKey: ['analytics-overview', timeframe],
    queryFn: () => api.get(`/analytics/overview?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  // Fetch trade analysis
  const { data: trades = [], isLoading: tradesLoading } = useQuery<TradeAnalysis[]>({
    queryKey: ['trade-analysis', timeframe],
    queryFn: () => api.get(`/analytics/trades?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  // Fetch performance metrics
  const { data: performanceMetrics = [], isLoading: metricsLoading } = useQuery<PerformanceMetric[]>({
    queryKey: ['performance-metrics', timeframe],
    queryFn: () => api.get(`/analytics/performance/metrics?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  // Fetch drawdown periods
  const { data: drawdownPeriods = [], isLoading: drawdownLoading } = useQuery<DrawdownPeriod[]>({
    queryKey: ['drawdown-periods', timeframe],
    queryFn: () => api.get(`/analytics/drawdown?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  // Fetch monthly returns
  const { data: monthlyReturns = [], isLoading: monthlyLoading } = useQuery<MonthlyReturns[]>({
    queryKey: ['monthly-returns', timeframe],
    queryFn: () => api.get(`/analytics/returns/monthly?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  // Fetch risk metrics
  const { data: riskMetrics, isLoading: riskLoading } = useQuery<RiskMetrics>({
    queryKey: ['risk-metrics', timeframe],
    queryFn: () => api.get(`/analytics/risk?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  if (overviewLoading) {
    return (
      <div className="p-6">
        <Loading size="lg" className="mx-auto" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Analytics & Performance
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Detailed analysis of your trading performance and risk metrics
          </p>
        </div>
        <div className="mt-4 lg:mt-0 flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
            {(['1M', '3M', '6M', '1Y', 'ALL'] as const).map((period) => (
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
          <Button variant="outline" size="sm">
            <DownloadIcon className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total Trades
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {overview?.totalTrades || 0}
              </p>
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-full">
              <ActivityIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Win Rate: <span className="font-medium text-green-600">{formatPercentage(overview?.winRate || 0)}</span>
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
                {formatCurrency(overview?.totalPnL || 0)}
              </p>
            </div>
            <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-full">
              <TrendingUpIcon className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Profit Factor: <span className="font-medium">{overview?.profitFactor?.toFixed(2) || 'N/A'}</span>
            </span>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Sharpe Ratio
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {overview?.sharpeRatio?.toFixed(2) || 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-full">
              <BarChart3Icon className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Volatility: <span className="font-medium">{formatPercentage(overview?.volatility || 0)}</span>
            </span>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Max Drawdown
              </p>
              <p className="text-2xl font-bold text-red-600">
                {formatPercentage(overview?.maxDrawdown || 0)}
              </p>
            </div>
            <div className="p-3 bg-red-100 dark:bg-red-900/20 rounded-full">
              <PieChartIcon className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Avg Loss: <span className="font-medium text-red-600">{formatCurrency(overview?.avgLoss || 0)}</span>
            </span>
          </div>
        </Card>
      </div>

      {/* Analysis Type Selector */}
      <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 rounded-lg p-1 w-fit">
        {[
          { key: 'performance', label: 'Performance', icon: TrendingUpIcon },
          { key: 'risk', label: 'Risk', icon: PieChartIcon },
          { key: 'trades', label: 'Trades', icon: ActivityIcon },
          { key: 'drawdown', label: 'Drawdown', icon: BarChart3Icon },
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setAnalysisType(key as any)}
            className={`flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              analysisType === key
                ? 'bg-blue-600 text-white'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
            }`}
          >
            <Icon className="w-4 h-4 mr-2" />
            {label}
          </button>
        ))}
      </div>

      {/* Main Content Based on Analysis Type */}
      {analysisType === 'performance' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Performance Chart */}
          <Card className="lg:col-span-2 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Cumulative Returns
              </h3>
              <div className="flex items-center space-x-2">
                <Button variant="ghost" size="sm">
                  <EyeIcon className="w-4 h-4" />
                </Button>
                <Button variant="ghost" size="sm">
                  <Settings2Icon className="w-4 h-4" />
                </Button>
              </div>
            </div>            <div className="h-64">
              <LineChart
                data={{
                  labels: performanceMetrics.map(metric => metric.name),
                  datasets: [
                    {
                      label: 'Performance Metrics',
                      data: performanceMetrics.map(metric => metric.value),
                      borderColor: 'rgb(59, 130, 246)',
                      backgroundColor: 'rgba(59, 130, 246, 0.1)',
                      tension: 0.3,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: false,
                    },
                    title: {
                      display: true,
                      text: 'Performance Trend',
                    },
                  },
                  scales: {
                    y: {
                      beginAtZero: false,
                    },
                  },
                }}
              />
            </div>
          </Card>

          {/* Performance Metrics */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
              Key Metrics
            </h3>
            {metricsLoading ? (
              <Loading size="sm" />
            ) : (
              <div className="space-y-4">
                {performanceMetrics.map((metric) => (
                  <div key={metric.name} className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {metric.name}
                    </span>
                    <div className="text-right">
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {metric.isPercentage ? formatPercentage(metric.value) : formatNumber(metric.value)}
                      </span>
                      {metric.benchmark && (
                        <p className="text-xs text-gray-500">
                          vs {metric.isPercentage ? formatPercentage(metric.benchmark) : formatNumber(metric.benchmark)}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* Monthly Returns Heatmap */}
          <Card className="lg:col-span-3 p-6">
            <div className="flex items-center justify-between mb-6">              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Monthly Returns
              </h3>
              <Badge variant="info">{monthlyReturns.length} months</Badge>
            </div>
            {monthlyLoading ? (
              <Loading size="sm" />
            ) : (
              <div className="space-y-6">
                {/* Monthly Returns Heatmap */}
                <div className="h-64">
                  <Heatmap
                    data={{
                      labels: monthlyReturns.map(month => month.month),
                      datasets: [
                        {
                          label: 'Monthly Returns',
                          data: monthlyReturns.map((month, index) => ({
                            x: index,
                            y: 0,
                            v: month.returns * 100
                          })),
                          backgroundColor: function(context) {
                            const value = context.parsed.v;
                            if (value > 5) return 'rgba(34, 197, 94, 0.8)';
                            if (value > 0) return 'rgba(34, 197, 94, 0.4)';
                            if (value > -5) return 'rgba(239, 68, 68, 0.4)';
                            return 'rgba(239, 68, 68, 0.8)';
                          },
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false,
                        },
                        title: {
                          display: true,
                          text: 'Monthly Returns Heatmap',
                        },
                        tooltip: {
                          callbacks: {
                            label: function(context) {
                              const month = monthlyReturns[context.dataIndex];
                              return `${month.month}: ${formatPercentage(month.returns)} (${month.trades} trades)`;
                            }
                          }
                        }
                      },
                    }}
                  />
                </div>
                
                {/* Monthly Returns Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {monthlyReturns.map((month) => (
                    <div
                      key={month.month}
                      className={`p-4 rounded-lg text-center ${
                        month.returns >= 0
                          ? 'bg-green-100 dark:bg-green-900/20'
                          : 'bg-red-100 dark:bg-red-900/20'
                      }`}
                    >
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {month.month}
                      </p>
                      <p
                        className={`text-lg font-bold ${
                          month.returns >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {formatPercentage(month.returns)}
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {month.trades} trades
                      </p>
                    </div>
                  ))}
                </div>
              </div>
          </Card>
        </div>
      )}

      {analysisType === 'risk' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Metrics */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
              Risk Metrics
            </h3>
            {riskLoading ? (
              <Loading size="sm" />
            ) : (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">VaR (95%)</span>
                  <span className="font-medium text-red-600">
                    {formatPercentage(riskMetrics?.var95 || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">VaR (99%)</span>
                  <span className="font-medium text-red-600">
                    {formatPercentage(riskMetrics?.var99 || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Expected Shortfall</span>
                  <span className="font-medium text-red-600">
                    {formatPercentage(riskMetrics?.expectedShortfall || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Beta</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {riskMetrics?.beta?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Alpha</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {formatPercentage(riskMetrics?.alpha || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Information Ratio</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {riskMetrics?.informationRatio?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Tracking Error</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {formatPercentage(riskMetrics?.trackingError || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Correlation</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {riskMetrics?.correlation?.toFixed(2) || 'N/A'}
                  </span>
                </div>
              </div>
            )}
          </Card>

          {/* Risk Chart */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
              Risk Distribution
            </h3>
            <div className="h-64 bg-gray-50 dark:bg-gray-800 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <PieChartIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 dark:text-gray-400">
                  Risk distribution chart
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {analysisType === 'trades' && (
        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Trade Analysis
            </h3>
            <div className="flex items-center space-x-2">
              <Badge variant="info">{trades.length} trades</Badge>
              <Button variant="outline" size="sm">
                <FilterIcon className="w-4 h-4 mr-2" />
                Filter
              </Button>
            </div>
          </div>
          {tradesLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead>Entry Price</TableHead>
                    <TableHead>Exit Price</TableHead>
                    <TableHead>Quantity</TableHead>
                    <TableHead>P&L</TableHead>
                    <TableHead>P&L %</TableHead>
                    <TableHead>Hold Period</TableHead>
                    <TableHead>Strategy</TableHead>
                    <TableHead>Entry Time</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {trades.map((trade) => (
                    <TableRow key={trade.id}>
                      <TableCell>
                        <span className="font-medium">{trade.symbol}</span>
                      </TableCell>
                      <TableCell>
                        <Badge variant={trade.side === 'buy' ? 'success' : 'error'}>
                          {trade.side.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>{formatCurrency(trade.entryPrice)}</TableCell>
                      <TableCell>{formatCurrency(trade.exitPrice)}</TableCell>
                      <TableCell>{formatNumber(trade.quantity)}</TableCell>
                      <TableCell>
                        <span
                          className={`font-medium ${
                            trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}
                        >
                          {formatCurrency(trade.pnl)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span
                          className={`font-medium ${
                            trade.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}
                        >
                          {formatPercentage(trade.pnlPercentage)}
                        </span>
                      </TableCell>
                      <TableCell>
                        {trade.holdingPeriod < 1440
                          ? `${Math.round(trade.holdingPeriod / 60)}h`
                          : `${Math.round(trade.holdingPeriod / 1440)}d`}
                      </TableCell>
                      <TableCell>
                        {trade.strategy && (
                          <Badge variant="secondary" size="sm">
                            {trade.strategy}
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        {new Date(trade.entryTime).toLocaleDateString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {trades.length === 0 && (
                <div className="text-center py-8">
                  <ActivityIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-600 dark:text-gray-400">No trades found</p>
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {analysisType === 'drawdown' && (
        <div className="space-y-6">
          {/* Drawdown Chart */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
              Drawdown Analysis
            </h3>            <div className="h-64">
              <AreaChart
                data={{
                  labels: drawdownPeriods.map(period => period.startDate),
                  datasets: [
                    {
                      label: 'Drawdown %',
                      data: drawdownPeriods.map(period => -Math.abs(period.maxDrawdown)),
                      borderColor: 'rgb(239, 68, 68)',
                      backgroundColor: 'rgba(239, 68, 68, 0.2)',
                      fill: true,
                      tension: 0.3,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: false,
                    },
                    title: {
                      display: true,
                      text: 'Drawdown Analysis',
                    },
                  },
                  scales: {
                    y: {
                      max: 0,
                      ticks: {
                        callback: function(value) {
                          return formatPercentage(Number(value));
                        }
                      }
                    },
                  },
                }}
              />
            </div>
          </Card>

          {/* Drawdown Periods */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Drawdown Periods
              </h3>
              <Badge variant="info">{drawdownPeriods.length} periods</Badge>
            </div>
            {drawdownLoading ? (
              <Loading size="sm" />
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Start Date</TableHead>
                      <TableHead>End Date</TableHead>
                      <TableHead>Duration (Days)</TableHead>
                      <TableHead>Max Drawdown</TableHead>
                      <TableHead>Recovery</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {drawdownPeriods.map((period, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          {new Date(period.startDate).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          {new Date(period.endDate).toLocaleDateString()}
                        </TableCell>
                        <TableCell>{period.duration}</TableCell>
                        <TableCell>
                          <span className="font-medium text-red-600">
                            {formatPercentage(period.maxDrawdown)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Badge variant={period.recovery ? 'success' : 'warning'}>
                            {period.recovery ? 'Recovered' : 'Ongoing'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {drawdownPeriods.length === 0 && (
                  <div className="text-center py-8">
                    <BarChart3Icon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-600 dark:text-gray-400">No significant drawdown periods</p>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
