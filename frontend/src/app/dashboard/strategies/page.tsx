'use client';

import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Loading } from '@/components/ui/Loading';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Table } from '@/components/ui/Table';
import { api } from '@/lib/api';
import { formatCurrency, formatPercentage, formatDate } from '@/lib/utils';

interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'technical' | 'quantitative' | 'machine_learning' | 'hybrid';
  status: 'draft' | 'testing' | 'live' | 'paused' | 'archived';
  version: string;
  creator: string;
  created_at: string;
  updated_at: string;
  performance: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades: number;
    avgHoldTime: number;
  };
  parameters: Record<string, any>;
  rules: StrategyRule[];
  backtest_results?: BacktestResult;
}

interface StrategyRule {
  id: string;
  type: 'entry' | 'exit' | 'risk_management' | 'position_sizing';
  condition: string;
  action: string;
  parameters: Record<string, any>;
  enabled: boolean;
  priority: number;
}

interface BacktestResult {
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  total_return: number;
  annual_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  equity_curve: Array<{ date: string; value: number }>;
  monthly_returns: Array<{ month: string; return: number }>;
}

interface Indicator {
  id: string;
  name: string;
  category: 'trend' | 'momentum' | 'volatility' | 'volume' | 'custom';
  description: string;
  parameters: Array<{
    name: string;
    type: 'number' | 'boolean' | 'string' | 'select';
    default: any;
    options?: any[];
    min?: number;
    max?: number;
  }>;
}

interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  tags: string[];
  template: Partial<Strategy>;
}

export default function StrategyBuilderPage() {
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showTemplateModal, setShowTemplateModal] = useState(false);
  const [showBacktestModal, setShowBacktestModal] = useState(false);
  const [builderMode, setBuilderMode] = useState<'visual' | 'code'>('visual');
  const [activeTab, setActiveTab] = useState<'overview' | 'rules' | 'parameters' | 'backtest'>('overview');

  const queryClient = useQueryClient();

  // Fetch strategies
  const { data: strategies, isLoading: strategiesLoading } = useQuery<Strategy[]>({
    queryKey: ['strategies'],
    queryFn: async () => {
      const response = await api.get('/strategies');
      return response.data;
    },
  });

  // Fetch indicators
  const { data: indicators, isLoading: indicatorsLoading } = useQuery<Indicator[]>({
    queryKey: ['indicators'],
    queryFn: async () => {
      const response = await api.get('/strategies/indicators');
      return response.data;
    },
  });

  // Fetch strategy templates
  const { data: templates, isLoading: templatesLoading } = useQuery<StrategyTemplate[]>({
    queryKey: ['strategy-templates'],
    queryFn: async () => {
      const response = await api.get('/strategies/templates');
      return response.data;
    },
  });

  // Create strategy mutation
  const createStrategyMutation = useMutation({
    mutationFn: async (strategy: Partial<Strategy>) => {
      const response = await api.post('/strategies', strategy);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      setShowCreateModal(false);
    },
  });

  // Update strategy mutation
  const updateStrategyMutation = useMutation({
    mutationFn: async ({ id, updates }: { id: string; updates: Partial<Strategy> }) => {
      const response = await api.put(`/strategies/${id}`, updates);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
  });

  // Run backtest mutation
  const backtestMutation = useMutation({
    mutationFn: async ({ strategyId, config }: { strategyId: string; config: any }) => {
      const response = await api.post(`/strategies/${strategyId}/backtest`, config);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      setShowBacktestModal(false);
    },
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'live': return 'text-green-600';
      case 'testing': return 'text-blue-600';
      case 'paused': return 'text-yellow-600';
      case 'draft': return 'text-gray-600';
      case 'archived': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'live': return 'success';
      case 'testing': return 'info';
      case 'paused': return 'warning';
      case 'draft': return 'default';
      case 'archived': return 'error';
      default: return 'default';
    }
  };

  const strategyColumns = [
    {
      header: 'Name',
      accessorKey: 'name',
      cell: ({ row }: any) => (
        <div>
          <div className="font-medium text-gray-900 dark:text-gray-100">
            {row.original.name}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-500">
            {row.original.type} • v{row.original.version}
          </div>
        </div>
      ),
    },
    {
      header: 'Status',
      accessorKey: 'status',
      cell: ({ row }: any) => (
        <Badge variant={getStatusBadgeVariant(row.original.status)}>
          {row.original.status}
        </Badge>
      ),
    },
    {
      header: 'Performance',
      accessorKey: 'performance',
      cell: ({ row }: any) => (
        <div className="text-sm">
          <div className="text-gray-900 dark:text-gray-100">
            {formatPercentage(row.original.performance?.totalReturn || 0)}
          </div>
          <div className="text-gray-500 dark:text-gray-500">
            Sharpe: {(row.original.performance?.sharpeRatio || 0).toFixed(2)}
          </div>
        </div>
      ),
    },
    {
      header: 'Win Rate',
      accessorKey: 'winRate',
      cell: ({ row }: any) => (
        <div className="text-sm">
          <div className="text-gray-900 dark:text-gray-100">
            {formatPercentage(row.original.performance?.winRate || 0)}
          </div>
          <div className="text-gray-500 dark:text-gray-500">
            {row.original.performance?.totalTrades || 0} trades
          </div>
        </div>
      ),
    },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: ({ row }: any) => (
        <div className="text-sm text-gray-500 dark:text-gray-500">
          {formatDate(row.original.created_at)}
        </div>
      ),
    },
    {
      header: 'Actions',
      id: 'actions',
      cell: ({ row }: any) => (
        <div className="flex space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSelectedStrategy(row.original)}
          >
            Edit
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSelectedStrategy(row.original);
              setShowBacktestModal(true);
            }}
          >
            Backtest
          </Button>
        </div>
      ),
    },
  ];

  if (strategiesLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loading size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Strategy Builder
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Create, test, and manage your trading strategies
          </p>
        </div>
        <div className="flex space-x-3">
          <Button variant="outline" onClick={() => setShowTemplateModal(true)}>
            Use Template
          </Button>
          <Button onClick={() => setShowCreateModal(true)}>
            Create Strategy
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="p-6">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Total Strategies
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {strategies?.length || 0}
          </div>
        </Card>

        <Card className="p-6">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Live Strategies
          </div>
          <div className="text-2xl font-bold text-green-600">
            {strategies?.filter(s => s.status === 'live').length || 0}
          </div>
        </Card>

        <Card className="p-6">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Best Performer
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {strategies && strategies.length > 0
              ? formatPercentage(Math.max(...strategies.map(s => s.performance?.totalReturn || 0)))
              : '0%'
            }
          </div>
        </Card>

        <Card className="p-6">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Avg Win Rate
          </div>
          <div className="text-2xl font-bold text-purple-600">
            {strategies && strategies.length > 0
              ? formatPercentage(
                  strategies.reduce((sum, s) => sum + (s.performance?.winRate || 0), 0) / strategies.length
                )
              : '0%'
            }
          </div>
        </Card>
      </div>

      {/* Strategy List */}
      {!selectedStrategy ? (
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Your Strategies
            </h3>
            <div className="flex space-x-2">
              <select className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md text-sm">
                <option>All Strategies</option>
                <option>Live</option>
                <option>Testing</option>
                <option>Draft</option>
              </select>
            </div>
          </div>
          
          {strategies && strategies.length > 0 ? (
            <Table
              data={strategies}
              columns={strategyColumns}
            />
          ) : (
            <div className="text-center py-12">
              <div className="text-gray-500 dark:text-gray-500 mb-4">
                No strategies found. Create your first strategy to get started.
              </div>
              <Button onClick={() => setShowCreateModal(true)}>
                Create Your First Strategy
              </Button>
            </div>
          )}
        </Card>
      ) : (
        /* Strategy Editor */
        <Card className="p-6">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center space-x-4">
              <Button
                variant="outline"
                onClick={() => setSelectedStrategy(null)}
              >
                ← Back
              </Button>
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  {selectedStrategy.name}
                </h2>
                <div className="flex items-center space-x-2 mt-1">
                  <Badge variant={getStatusBadgeVariant(selectedStrategy.status)}>
                    {selectedStrategy.status}
                  </Badge>
                  <span className="text-sm text-gray-500 dark:text-gray-500">
                    v{selectedStrategy.version}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex space-x-3">
              <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
                <button
                  onClick={() => setBuilderMode('visual')}
                  className={`px-3 py-1 text-sm rounded-md transition-colors ${
                    builderMode === 'visual'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}
                >
                  Visual
                </button>
                <button
                  onClick={() => setBuilderMode('code')}
                  className={`px-3 py-1 text-sm rounded-md transition-colors ${
                    builderMode === 'code'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}
                >
                  Code
                </button>
              </div>
              <Button variant="outline">
                Test Strategy
              </Button>
              <Button>
                Deploy Live
              </Button>
            </div>
          </div>

          {/* Strategy Editor Tabs */}
          <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'overview', label: 'Overview' },
                { id: 'rules', label: 'Trading Rules' },
                { id: 'parameters', label: 'Parameters' },
                { id: 'backtest', label: 'Backtest Results' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Strategy Information
                </h4>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Name
                    </label>
                    <Input value={selectedStrategy.name} />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Description
                    </label>
                    <textarea
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                      rows={3}
                      value={selectedStrategy.description}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Type
                    </label>
                    <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md">
                      <option value="technical">Technical Analysis</option>
                      <option value="quantitative">Quantitative</option>
                      <option value="machine_learning">Machine Learning</option>
                      <option value="hybrid">Hybrid</option>
                    </select>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Performance Metrics
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Total Return</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {formatPercentage(selectedStrategy.performance?.totalReturn || 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {(selectedStrategy.performance?.sharpeRatio || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</span>
                    <span className="font-medium text-red-600">
                      {formatPercentage(selectedStrategy.performance?.maxDrawdown || 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Win Rate</span>
                    <span className="font-medium text-green-600">
                      {formatPercentage(selectedStrategy.performance?.winRate || 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Total Trades</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {selectedStrategy.performance?.totalTrades || 0}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'rules' && (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Trading Rules
                </h4>
                <Button size="sm">
                  Add Rule
                </Button>
              </div>
              
              {builderMode === 'visual' ? (
                <div className="space-y-4">
                  {selectedStrategy.rules?.map((rule, index) => (
                    <div
                      key={rule.id}
                      className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                    >
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center space-x-2">
                          <Badge variant="info">{rule.type}</Badge>
                          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            Rule #{index + 1}
                          </span>
                        </div>
                        <div className="flex space-x-2">
                          <Button variant="outline" size="sm">
                            Edit
                          </Button>
                          <Button variant="outline" size="sm">
                            Delete
                          </Button>
                        </div>
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        <strong>Condition:</strong> {rule.condition}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        <strong>Action:</strong> {rule.action}
                      </div>
                    </div>
                  )) || (
                    <div className="text-center py-8 text-gray-500 dark:text-gray-500">
                      No rules defined. Add your first trading rule to get started.
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
                  <div className="mb-2 text-gray-400"># Entry Rules</div>
                  <div>if RSI(14) &lt; 30 and MACD_signal == 'bullish':</div>
                  <div className="ml-4">buy(size=calculate_position_size())</div>
                  <br />
                  <div className="mb-2 text-gray-400"># Exit Rules</div>
                  <div>if RSI(14) &gt; 70 or stop_loss_triggered():</div>
                  <div className="ml-4">sell(position='all')</div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'parameters' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Indicator Parameters
                </h4>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      RSI Period
                    </label>
                    <Input type="number" value="14" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      MACD Fast Period
                    </label>
                    <Input type="number" value="12" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      MACD Slow Period
                    </label>
                    <Input type="number" value="26" />
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Risk Management
                </h4>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Stop Loss (%)
                    </label>
                    <Input type="number" value="2" step="0.1" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Take Profit (%)
                    </label>
                    <Input type="number" value="4" step="0.1" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Position Size (%)
                    </label>
                    <Input type="number" value="5" step="0.1" />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'backtest' && (
            <div>
              <div className="flex justify-between items-center mb-6">
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Backtest Results
                </h4>
                <Button onClick={() => setShowBacktestModal(true)}>
                  Run New Backtest
                </Button>
              </div>

              {selectedStrategy.backtest_results ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="p-4">
                    <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-4">
                      Performance Summary
                    </h5>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Initial Capital</span>
                        <span className="font-medium">
                          {formatCurrency(selectedStrategy.backtest_results.initial_capital)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Final Capital</span>
                        <span className="font-medium">
                          {formatCurrency(selectedStrategy.backtest_results.final_capital)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Total Return</span>
                        <span className="font-medium text-green-600">
                          {formatPercentage(selectedStrategy.backtest_results.total_return)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Annual Return</span>
                        <span className="font-medium">
                          {formatPercentage(selectedStrategy.backtest_results.annual_return)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</span>
                        <span className="font-medium">
                          {selectedStrategy.backtest_results.sharpe_ratio.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</span>
                        <span className="font-medium text-red-600">
                          {formatPercentage(selectedStrategy.backtest_results.max_drawdown)}
                        </span>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4">
                    <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-4">
                      Trade Statistics
                    </h5>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Total Trades</span>
                        <span className="font-medium">
                          {selectedStrategy.backtest_results.total_trades}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Win Rate</span>
                        <span className="font-medium text-green-600">
                          {formatPercentage(selectedStrategy.backtest_results.win_rate)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Profit Factor</span>
                        <span className="font-medium">
                          {selectedStrategy.backtest_results.profit_factor.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  </Card>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-gray-500 dark:text-gray-500 mb-4">
                    No backtest results available. Run a backtest to see performance metrics.
                  </div>
                  <Button onClick={() => setShowBacktestModal(true)}>
                    Run Backtest
                  </Button>
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {/* Create Strategy Modal */}
      {showCreateModal && (
        <Modal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          title="Create New Strategy"
          size="lg"
        >
          <div className="p-4">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Strategy Name
                </label>
                <Input placeholder="Enter strategy name" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Description
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                  rows={3}
                  placeholder="Describe your strategy"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Strategy Type
                </label>
                <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md">
                  <option value="technical">Technical Analysis</option>
                  <option value="quantitative">Quantitative</option>
                  <option value="machine_learning">Machine Learning</option>
                  <option value="hybrid">Hybrid</option>
                </select>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <Button variant="outline" onClick={() => setShowCreateModal(false)}>
                Cancel
              </Button>
              <Button onClick={() => setShowCreateModal(false)}>
                Create Strategy
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {/* Template Modal */}
      {showTemplateModal && (
        <Modal
          isOpen={showTemplateModal}
          onClose={() => setShowTemplateModal(false)}
          title="Choose Strategy Template"
          size="xl"
        >
          <div className="p-4">
            {templatesLoading ? (
              <Loading size="lg" />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {templates?.map((template) => (
                  <div
                    key={template.id}
                    className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-medium text-gray-900 dark:text-gray-100">
                        {template.name}
                      </h4>
                      <Badge variant="info">{template.difficulty}</Badge>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {template.description}
                    </p>
                    <div className="flex flex-wrap gap-1 mb-3">
                      {template.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-xs rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <Button size="sm" className="w-full">
                      Use Template
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
}
