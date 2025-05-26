'use client';

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  PieChartIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  DollarSignIcon,
  BarChart3Icon,
  ArrowUpIcon,
  ArrowDownIcon,
  CalendarIcon,
  FilterIcon,
  DownloadIcon,
  RefreshCwIcon,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Loading } from '@/components/ui/Loading';
import { Badge } from '@/components/ui/Badge';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table';
import { api } from '@/lib/api';
import { formatCurrency, formatNumber, formatPercentage } from '@/lib/utils';
import { AreaChart, DoughnutChart, BarChart } from '@/components/charts';

interface PortfolioSummary {
  totalValue: number;
  totalCost: number;
  totalPnL: number;
  totalPnLPercentage: number;
  dailyPnL: number;
  dailyPnLPercentage: number;
  cash: number;
  marginUsed: number;
  marginAvailable: number;
  buyingPower: number;
}

interface PortfolioHolding {
  symbol: string;
  name: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  marketValue: number;
  costBasis: number;
  pnl: number;
  pnlPercentage: number;
  dailyChange: number;
  dailyChangePercentage: number;
  weight: number;
  sector: string;
  dividendYield?: number;
}

interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
}

interface SectorAllocation {
  sector: string;
  value: number;
  percentage: number;
  pnl: number;
  pnlPercentage: number;
}

export default function PortfolioPage() {
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | 'ALL'>('1M');
  const [sortBy, setSortBy] = useState<'symbol' | 'value' | 'pnl' | 'weight'>('value');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Fetch portfolio summary
  const { data: summary, isLoading: summaryLoading } = useQuery<PortfolioSummary>({
    queryKey: ['portfolio-summary'],
    queryFn: () => api.get('/trading/portfolio/summary'),
    refetchInterval: 30000,
  });

  // Fetch portfolio holdings
  const { data: holdings = [], isLoading: holdingsLoading, refetch: refetchHoldings } = useQuery<PortfolioHolding[]>({
    queryKey: ['portfolio-holdings', sortBy, sortOrder],
    queryFn: () => api.get(`/trading/portfolio/holdings?sortBy=${sortBy}&sortOrder=${sortOrder}`),
    refetchInterval: 30000,
  });

  // Fetch performance metrics
  const { data: metrics, isLoading: metricsLoading } = useQuery<PerformanceMetrics>({
    queryKey: ['portfolio-metrics', timeframe],
    queryFn: () => api.get(`/analytics/portfolio/metrics?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  // Fetch sector allocation
  const { data: sectorAllocation = [], isLoading: sectorLoading } = useQuery<SectorAllocation[]>({
    queryKey: ['portfolio-sectors'],
    queryFn: () => api.get('/trading/portfolio/sectors'),
    refetchInterval: 60000,
  });

  // Fetch performance chart data
  const { data: performanceData, isLoading: performanceLoading } = useQuery({
    queryKey: ['portfolio-performance-chart', timeframe],
    queryFn: () => api.get(`/analytics/portfolio/performance?timeframe=${timeframe}`),
    refetchInterval: 60000,
  });

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  if (summaryLoading) {
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
            Portfolio Management
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Monitor your holdings and portfolio performance
          </p>
        </div>
        <div className="mt-4 lg:mt-0 flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
            {(['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL'] as const).map((period) => (
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
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetchHoldings()}
          >
            <RefreshCwIcon className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total Value
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(summary?.totalValue || 0)}
              </p>
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-full">
              <DollarSignIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
          <div className="mt-4 flex items-center">
            {(summary?.dailyPnL || 0) >= 0 ? (
              <ArrowUpIcon className="w-4 h-4 text-green-500 mr-1" />
            ) : (
              <ArrowDownIcon className="w-4 h-4 text-red-500 mr-1" />
            )}
            <span
              className={`text-sm font-medium ${
                (summary?.dailyPnL || 0) >= 0 ? 'text-green-600' : 'text-red-600'
              }`}
            >
              {formatCurrency(summary?.dailyPnL || 0)} ({formatPercentage(summary?.dailyPnLPercentage || 0)}) today
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
                {formatCurrency(summary?.totalPnL || 0)}
              </p>
            </div>
            <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-full">
              <TrendingUpIcon className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
          </div>
          <div className="mt-4">
            <span
              className={`text-sm font-medium ${
                (summary?.totalPnLPercentage || 0) >= 0 ? 'text-green-600' : 'text-red-600'
              }`}
            >
              {formatPercentage(summary?.totalPnLPercentage || 0)} return
            </span>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Cash Balance
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(summary?.cash || 0)}
              </p>
            </div>
            <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-full">
              <BarChart3Icon className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Buying Power: {formatCurrency(summary?.buyingPower || 0)}
            </span>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Margin Used
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(summary?.marginUsed || 0)}
              </p>
            </div>
            <div className="p-3 bg-orange-100 dark:bg-orange-900/20 rounded-full">
              <PieChartIcon className="w-6 h-6 text-orange-600 dark:text-orange-400" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Available: {formatCurrency(summary?.marginAvailable || 0)}
            </span>
          </div>
        </Card>
      </div>

      {/* Performance Metrics & Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance Chart */}
        <Card className="lg:col-span-2 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Portfolio Performance
            </h3>
            <div className="flex items-center space-x-2">
              <CalendarIcon className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-600 dark:text-gray-400">{timeframe}</span>
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
                    label: 'Portfolio Performance',
                    data: [0, 5.2, -2.1, 12.8, 8.6, (metrics?.totalReturn || 0) * 100],
                    borderColor: '#10B981',
                    backgroundColor: '#10B98140',
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

        {/* Performance Metrics */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
            Performance Metrics
          </h3>
          {metricsLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Total Return</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {formatPercentage(metrics?.totalReturn || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Annualized Return</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {formatPercentage(metrics?.annualizedReturn || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Volatility</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {formatPercentage(metrics?.volatility || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {metrics?.sharpeRatio?.toFixed(2) || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</span>
                <span className="font-medium text-red-600">
                  {formatPercentage(metrics?.maxDrawdown || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Win Rate</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {formatPercentage(metrics?.winRate || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Profit Factor</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {metrics?.profitFactor?.toFixed(2) || 'N/A'}
                </span>
              </div>
            </div>
          )}
        </Card>      </div>

      {/* Sector Allocation Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
            Sector Allocation Chart
          </h3>
          <div className="h-80">
            {sectorLoading ? (
              <div className="flex items-center justify-center h-full">
                <Loading size="md" />
              </div>
            ) : (
              <DoughnutChart
                data={{
                  labels: sectorAllocation.map(sector => sector.sector),
                  datasets: [
                    {
                      data: sectorAllocation.map(sector => sector.percentage),
                      backgroundColor: [
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(139, 92, 246, 0.8)',
                        'rgba(236, 72, 153, 0.8)',
                        'rgba(34, 197, 94, 0.8)',
                        'rgba(251, 191, 36, 0.8)',
                      ],
                      borderColor: [
                        'rgb(59, 130, 246)',
                        'rgb(16, 185, 129)',
                        'rgb(245, 158, 11)',
                        'rgb(239, 68, 68)',
                        'rgb(139, 92, 246)',
                        'rgb(236, 72, 153)',
                        'rgb(34, 197, 94)',
                        'rgb(251, 191, 36)',
                      ],
                      borderWidth: 2,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'right',
                    },
                    title: {
                      display: true,
                      text: 'Portfolio Sector Distribution',
                    },
                    tooltip: {
                      callbacks: {
                        label: function(context) {
                          const sector = sectorAllocation[context.dataIndex];
                          return `${context.label}: ${formatPercentage(context.parsed)} (${formatCurrency(sector.value)})`;
                        }
                      }
                    }
                  },
                }}
              />
            )}
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
            Top Holdings
          </h3>
          <div className="space-y-4">
            {holdings.slice(0, 8).map((holding, index) => (
              <div key={holding.symbol} className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center mr-3">
                    <span className="text-xs font-bold text-blue-600 dark:text-blue-400">
                      {index + 1}
                    </span>
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {holding.symbol}
                    </p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {formatPercentage(holding.weight)} of portfolio
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-medium text-gray-900 dark:text-gray-100">
                    {formatCurrency(holding.marketValue)}
                  </p>
                  <p className={`text-xs ${holding.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPercentage(holding.pnlPercentage)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Sector Allocation */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Sector Allocation
          </h3>
          <Button variant="outline" size="sm">
            <FilterIcon className="w-4 h-4 mr-2" />
            Filter
          </Button>
        </div>
        {sectorLoading ? (
          <Loading size="sm" />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {sectorAllocation.map((sector) => (
              <div
                key={sector.sector}
                className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {sector.sector}
                  </span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {formatPercentage(sector.percentage)}
                  </span>
                </div>
                <p className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-1">
                  {formatCurrency(sector.value)}
                </p>
                <p
                  className={`text-sm ${
                    sector.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {formatCurrency(sector.pnl)} ({formatPercentage(sector.pnlPercentage)})
                </p>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Holdings Table */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Portfolio Holdings
          </h3>
          <div className="flex items-center space-x-2">
            <Badge variant="info">{holdings.length} holdings</Badge>
            <Button variant="outline" size="sm">
              <FilterIcon className="w-4 h-4 mr-2" />
              Filter
            </Button>
          </div>
        </div>
        {holdingsLoading ? (
          <Loading size="sm" />
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead
                    sortable
                    sortDirection={sortBy === 'symbol' ? sortOrder : null}
                    onSort={() => handleSort('symbol')}
                  >
                    Symbol
                  </TableHead>
                  <TableHead>Quantity</TableHead>
                  <TableHead>Avg Price</TableHead>
                  <TableHead>Current Price</TableHead>
                  <TableHead
                    sortable
                    sortDirection={sortBy === 'value' ? sortOrder : null}
                    onSort={() => handleSort('value')}
                  >
                    Market Value
                  </TableHead>
                  <TableHead
                    sortable
                    sortDirection={sortBy === 'pnl' ? sortOrder : null}
                    onSort={() => handleSort('pnl')}
                  >
                    P&L
                  </TableHead>
                  <TableHead>P&L %</TableHead>
                  <TableHead>Daily Change</TableHead>
                  <TableHead
                    sortable
                    sortDirection={sortBy === 'weight' ? sortOrder : null}
                    onSort={() => handleSort('weight')}
                  >
                    Weight
                  </TableHead>
                  <TableHead>Sector</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {holdings.map((holding) => (
                  <TableRow key={holding.symbol}>
                    <TableCell>
                      <div>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {holding.symbol}
                        </span>
                        <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                          {holding.name}
                        </p>
                      </div>
                    </TableCell>
                    <TableCell>{formatNumber(holding.quantity)}</TableCell>
                    <TableCell>{formatCurrency(holding.avgPrice)}</TableCell>
                    <TableCell>{formatCurrency(holding.currentPrice)}</TableCell>
                    <TableCell>
                      <span className="font-medium">
                        {formatCurrency(holding.marketValue)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span
                        className={`font-medium ${
                          holding.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {formatCurrency(holding.pnl)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span
                        className={`font-medium ${
                          holding.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {formatPercentage(holding.pnlPercentage)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        {holding.dailyChangePercentage >= 0 ? (
                          <ArrowUpIcon className="w-3 h-3 text-green-500 mr-1" />
                        ) : (
                          <ArrowDownIcon className="w-3 h-3 text-red-500 mr-1" />
                        )}
                        <span
                          className={`text-sm ${
                            holding.dailyChangePercentage >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}
                        >
                          {formatPercentage(holding.dailyChangePercentage)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${Math.min(holding.weight * 100, 100)}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {formatPercentage(holding.weight)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" size="sm">
                        {holding.sector}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {holdings.length === 0 && (
              <div className="text-center py-8">
                <PieChartIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-600 dark:text-gray-400">No holdings found</p>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
