'use client';

import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  TrendingUpIcon,
  TrendingDownIcon,
  DollarSignIcon,
  BarChart3Icon,
  PlusIcon,
  RefreshCwIcon,
  SettingsIcon,
  AlertCircleIcon,
  CheckCircleIcon,
  XCircleIcon,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Loading } from '@/components/ui/Loading';
import { Badge } from '@/components/ui/Badge';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell, TableActions } from '@/components/ui/Table';
import { Modal, ModalHeader, ModalBody, ModalFooter } from '@/components/ui/Modal';
import { api } from '@/lib/api';
import { useMarketDataStore, useTradingStore } from '@/lib/store';
import { formatCurrency, formatNumber, formatPercentage } from '@/lib/utils';
import { CandlestickChart, BarChart } from '@/components/charts';

interface OrderForm {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: string;
  price?: string;
  stopPrice?: string;
  timeInForce: 'day' | 'gtc' | 'fok' | 'ioc';
}

interface Position {
  id: string;
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercentage: number;
  value: number;
  side: 'long' | 'short';
  createdAt: string;
}

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected';
  timeInForce: 'day' | 'gtc' | 'fok' | 'ioc';
  createdAt: string;
  updatedAt: string;
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercentage: number;
  volume: number;
  bid: number;
  ask: number;
  high: number;
  low: number;
}

export default function TradingPage() {
  const queryClient = useQueryClient();
  const { marketData } = useMarketDataStore();
  const [showOrderModal, setShowOrderModal] = useState(false);
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [showClosePositionModal, setShowClosePositionModal] = useState(false);
  const [orderForm, setOrderForm] = useState<OrderForm>({
    symbol: '',
    side: 'buy',
    type: 'market',
    quantity: '',
    timeInForce: 'day',
  });

  // Fetch positions
  const { data: positions = [], isLoading: positionsLoading, refetch: refetchPositions } = useQuery<Position[]>({
    queryKey: ['positions'],
    queryFn: () => api.get('/trading/positions'),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch orders
  const { data: orders = [], isLoading: ordersLoading, refetch: refetchOrders } = useQuery<Order[]>({
    queryKey: ['orders'],
    queryFn: () => api.get('/trading/orders'),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch watchlist symbols
  const { data: watchlist = [], isLoading: watchlistLoading } = useQuery<MarketData[]>({
    queryKey: ['watchlist'],
    queryFn: () => api.get('/data/watchlist'),
    refetchInterval: 2000, // Refresh every 2 seconds
  });

  // Place order mutation
  const placeOrderMutation = useMutation({
    mutationFn: (orderData: any) => api.post('/trading/orders', orderData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      setShowOrderModal(false);
      resetOrderForm();
    },
  });

  // Cancel order mutation
  const cancelOrderMutation = useMutation({
    mutationFn: (orderId: string) => api.delete(`/trading/orders/${orderId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });

  // Close position mutation
  const closePositionMutation = useMutation({
    mutationFn: (positionId: string) => api.post(`/trading/positions/${positionId}/close`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      setShowClosePositionModal(false);
      setSelectedPosition(null);
    },
  });

  const resetOrderForm = () => {
    setOrderForm({
      symbol: '',
      side: 'buy',
      type: 'market',
      quantity: '',
      timeInForce: 'day',
    });
  };

  const handlePlaceOrder = (e: React.FormEvent) => {
    e.preventDefault();
    
    const orderData = {
      symbol: orderForm.symbol.toUpperCase(),
      side: orderForm.side,
      type: orderForm.type,
      quantity: parseFloat(orderForm.quantity),
      timeInForce: orderForm.timeInForce,
      ...(orderForm.type !== 'market' && orderForm.price && { price: parseFloat(orderForm.price) }),
      ...(orderForm.type === 'stop' || orderForm.type === 'stop_limit') && orderForm.stopPrice && { stopPrice: parseFloat(orderForm.stopPrice) },
    };

    placeOrderMutation.mutate(orderData);
  };

  const handleClosePosition = () => {
    if (selectedPosition) {
      closePositionMutation.mutate(selectedPosition.id);
    }
  };

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      pending: { variant: 'warning' as const, text: 'Pending' },
      filled: { variant: 'success' as const, text: 'Filled' },
      partial: { variant: 'info' as const, text: 'Partial' },
      cancelled: { variant: 'secondary' as const, text: 'Cancelled' },
      rejected: { variant: 'error' as const, text: 'Rejected' },
    };
    
    const config = statusConfig[status as keyof typeof statusConfig] || { variant: 'default' as const, text: status };
    return <Badge variant={config.variant}>{config.text}</Badge>;
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Trading Interface
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Manage your positions and execute trades
          </p>
        </div>
        <div className="mt-4 lg:mt-0 flex items-center space-x-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              refetchPositions();
              refetchOrders();
            }}
          >
            <RefreshCwIcon className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => setShowOrderModal(true)}
          >
            <PlusIcon className="w-4 h-4 mr-2" />
            New Order
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Open Positions
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {positions.length}
              </p>
            </div>
            <BarChart3Icon className="w-8 h-8 text-blue-600" />
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Active Orders
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {orders.filter(order => order.status === 'pending').length}
              </p>
            </div>
            <TrendingUpIcon className="w-8 h-8 text-green-600" />
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total P&L
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(positions.reduce((sum, pos) => sum + pos.pnl, 0))}
              </p>
            </div>
            <DollarSignIcon className="w-8 h-8 text-purple-600" />
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Portfolio Value
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {formatCurrency(positions.reduce((sum, pos) => sum + pos.value, 0))}
              </p>
            </div>
            <TrendingDownIcon className="w-8 h-8 text-orange-600" />
          </div>        </Card>
      </div>

      {/* Price Chart Section */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Price Chart - {orderForm.symbol || 'Select Symbol'}
          </h3>
          <div className="flex space-x-2">
            <Button variant="ghost" size="sm">1D</Button>
            <Button variant="ghost" size="sm">1W</Button>
            <Button variant="ghost" size="sm">1M</Button>
            <Button variant="ghost" size="sm">3M</Button>
          </div>        </div>
        <div className="h-80">
          <CandlestickChart
            data={{
              labels: Array.from({ length: 30 }, (_, i) => {
                const date = new Date();
                date.setDate(date.getDate() - (29 - i));
                return date.toLocaleDateString();
              }),
              datasets: [
                {
                  label: orderForm.symbol || 'Symbol',
                  data: Array.from({ length: 30 }, () => ({
                    x: Date.now() + Math.random() * 1000000,
                    o: 100 + Math.random() * 50,
                    h: 120 + Math.random() * 30,
                    l: 90 + Math.random() * 20,
                    c: 110 + Math.random() * 40,
                    v: Math.floor(Math.random() * 1000000)
                  })),
                  borderColor: 'rgb(34, 197, 94)',
                  backgroundColor: 'rgba(34, 197, 94, 0.1)',
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                title: {
                  display: true,
                  text: `${orderForm.symbol || 'Symbol'} Price Action`,
                },
              },
            }}
          />
        </div>
      </Card>

      {/* Volume Chart */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Volume Analysis - {orderForm.symbol || 'Select Symbol'}
          </h3>
        </div>
        <div className="h-48">
          <BarChart
            data={{
              labels: Array.from({ length: 30 }, (_, i) => {
                const date = new Date();
                date.setDate(date.getDate() - (29 - i));
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
              }),
              datasets: [
                {
                  label: 'Volume',
                  data: Array.from({ length: 30 }, () => Math.floor(Math.random() * 2000000) + 500000),
                  backgroundColor: Array.from({ length: 30 }, () => 
                    Math.random() > 0.5 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
                  ),
                  borderColor: Array.from({ length: 30 }, () => 
                    Math.random() > 0.5 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)'
                  ),
                  borderWidth: 1,
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
                  text: 'Daily Trading Volume',
                },
              },
              scales: {
                y: {
                  beginAtZero: true,
                  ticks: {
                    callback: function(value) {
                      return formatNumber(Number(value), { compact: true });
                    }
                  }
                },
              },
            }}
          />
        </div>
      </Card>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Watchlist */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Watchlist
            </h3>
            <Button variant="ghost" size="sm">
              <SettingsIcon className="w-4 h-4" />
            </Button>
          </div>
          {watchlistLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-3">
              {watchlist.map((item) => (
                <div
                  key={item.symbol}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                  onClick={() => setOrderForm({ ...orderForm, symbol: item.symbol })}
                >
                  <div>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {item.symbol}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Vol: {formatNumber(item.volume, { compact: true })}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {formatCurrency(item.price)}
                    </p>
                    <p
                      className={`text-sm ${
                        item.changePercentage >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {formatPercentage(item.changePercentage)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Positions & Orders */}
        <div className="lg:col-span-3 space-y-6">
          {/* Positions */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Open Positions
              </h3>
              <Badge variant="info">{positions.length} positions</Badge>
            </div>
            {positionsLoading ? (
              <Loading size="sm" />
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead>Quantity</TableHead>
                      <TableHead>Avg Price</TableHead>
                      <TableHead>Current Price</TableHead>
                      <TableHead>P&L</TableHead>
                      <TableHead>P&L %</TableHead>
                      <TableHead>Value</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {positions.map((position) => (
                      <TableRow key={position.id}>
                        <TableCell>
                          <span className="font-medium">{position.symbol}</span>
                        </TableCell>
                        <TableCell>
                          <Badge variant={position.side === 'long' ? 'success' : 'error'}>
                            {position.side.toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell>{formatNumber(position.quantity)}</TableCell>
                        <TableCell>{formatCurrency(position.avgPrice)}</TableCell>
                        <TableCell>{formatCurrency(position.currentPrice)}</TableCell>
                        <TableCell>
                          <span
                            className={`font-medium ${
                              position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}
                          >
                            {formatCurrency(position.pnl)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span
                            className={`font-medium ${
                              position.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}
                          >
                            {formatPercentage(position.pnlPercentage)}
                          </span>
                        </TableCell>
                        <TableCell>{formatCurrency(position.value)}</TableCell>
                        <TableActions>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setSelectedPosition(position);
                              setShowClosePositionModal(true);
                            }}
                          >
                            Close
                          </Button>
                        </TableActions>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {positions.length === 0 && (
                  <div className="text-center py-8">
                    <BarChart3Icon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-600 dark:text-gray-400">No open positions</p>
                  </div>
                )}
              </div>
            )}
          </Card>

          {/* Orders */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Recent Orders
              </h3>
              <Badge variant="info">{orders.length} orders</Badge>
            </div>
            {ordersLoading ? (
              <Loading size="sm" />
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Quantity</TableHead>
                      <TableHead>Price</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Time</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {orders.map((order) => (
                      <TableRow key={order.id}>
                        <TableCell>
                          <span className="font-medium">{order.symbol}</span>
                        </TableCell>
                        <TableCell>
                          <Badge variant={order.side === 'buy' ? 'success' : 'error'}>
                            {order.side.toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span className="capitalize">{order.type}</span>
                        </TableCell>
                        <TableCell>{formatNumber(order.quantity)}</TableCell>
                        <TableCell>
                          {order.price ? formatCurrency(order.price) : 'Market'}
                        </TableCell>
                        <TableCell>{getStatusBadge(order.status)}</TableCell>
                        <TableCell>
                          {new Date(order.createdAt).toLocaleTimeString()}
                        </TableCell>
                        <TableActions>
                          {order.status === 'pending' && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => cancelOrderMutation.mutate(order.id)}
                              disabled={cancelOrderMutation.isPending}
                            >
                              Cancel
                            </Button>
                          )}
                        </TableActions>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {orders.length === 0 && (
                  <div className="text-center py-8">
                    <AlertCircleIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-600 dark:text-gray-400">No recent orders</p>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* New Order Modal */}
      <Modal
        isOpen={showOrderModal}
        onClose={() => setShowOrderModal(false)}
        title="Place New Order"
        size="md"
      >
        <form onSubmit={handlePlaceOrder}>
          <ModalBody>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Symbol
                  </label>
                  <Input
                    type="text"
                    value={orderForm.symbol}
                    onChange={(e) => setOrderForm({ ...orderForm, symbol: e.target.value.toUpperCase() })}
                    placeholder="AAPL"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Side
                  </label>
                  <select
                    value={orderForm.side}
                    onChange={(e) => setOrderForm({ ...orderForm, side: e.target.value as 'buy' | 'sell' })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  >
                    <option value="buy">Buy</option>
                    <option value="sell">Sell</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Order Type
                  </label>
                  <select
                    value={orderForm.type}
                    onChange={(e) => setOrderForm({ ...orderForm, type: e.target.value as any })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  >
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                    <option value="stop">Stop</option>
                    <option value="stop_limit">Stop Limit</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Quantity
                  </label>
                  <Input
                    type="number"
                    value={orderForm.quantity}
                    onChange={(e) => setOrderForm({ ...orderForm, quantity: e.target.value })}
                    placeholder="100"
                    required
                    min="0"
                    step="any"
                  />
                </div>
              </div>

              {orderForm.type !== 'market' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Price
                  </label>
                  <Input
                    type="number"
                    value={orderForm.price || ''}
                    onChange={(e) => setOrderForm({ ...orderForm, price: e.target.value })}
                    placeholder="150.00"
                    required={orderForm.type !== 'market'}
                    min="0"
                    step="0.01"
                  />
                </div>
              )}

              {(orderForm.type === 'stop' || orderForm.type === 'stop_limit') && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Stop Price
                  </label>
                  <Input
                    type="number"
                    value={orderForm.stopPrice || ''}
                    onChange={(e) => setOrderForm({ ...orderForm, stopPrice: e.target.value })}
                    placeholder="145.00"
                    required
                    min="0"
                    step="0.01"
                  />
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Time in Force
                </label>
                <select
                  value={orderForm.timeInForce}
                  onChange={(e) => setOrderForm({ ...orderForm, timeInForce: e.target.value as any })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                >
                  <option value="day">Day</option>
                  <option value="gtc">Good Till Cancelled</option>
                  <option value="fok">Fill or Kill</option>
                  <option value="ioc">Immediate or Cancel</option>
                </select>
              </div>
            </div>
          </ModalBody>

          <ModalFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShowOrderModal(false)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              disabled={placeOrderMutation.isPending}
            >
              {placeOrderMutation.isPending ? 'Placing...' : 'Place Order'}
            </Button>
          </ModalFooter>
        </form>
      </Modal>

      {/* Close Position Modal */}
      <Modal
        isOpen={showClosePositionModal}
        onClose={() => setShowClosePositionModal(false)}
        title="Close Position"
        size="sm"
      >
        <ModalBody>
          {selectedPosition && (
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
                  {selectedPosition.symbol}
                </h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Quantity</p>
                    <p className="font-medium">{formatNumber(selectedPosition.quantity)}</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Avg Price</p>
                    <p className="font-medium">{formatCurrency(selectedPosition.avgPrice)}</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Current P&L</p>
                    <p
                      className={`font-medium ${
                        selectedPosition.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {formatCurrency(selectedPosition.pnl)}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Current Value</p>
                    <p className="font-medium">{formatCurrency(selectedPosition.value)}</p>
                  </div>
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Are you sure you want to close this position? This action cannot be undone.
              </p>
            </div>
          )}
        </ModalBody>

        <ModalFooter>
          <Button
            variant="outline"
            onClick={() => setShowClosePositionModal(false)}
          >
            Cancel
          </Button>
          <Button
            variant="error"
            onClick={handleClosePosition}
            disabled={closePositionMutation.isPending}
          >
            {closePositionMutation.isPending ? 'Closing...' : 'Close Position'}
          </Button>
        </ModalFooter>
      </Modal>
    </div>
  );
}
