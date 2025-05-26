'use client';

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Loading } from '@/components/ui/Loading';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { api } from '@/lib/api';
import { formatCurrency, formatPercentage, formatDate } from '@/lib/utils';
import { RiskGauge, DoughnutChart, AreaChart } from '@/components/charts';

interface RiskMetrics {
  var95: number;
  var99: number;
  expectedShortfall: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  volatility: number;
  beta: number;
  riskScore: number;
  marginUtilization: number;
  leverageRatio: number;
  portfolioValue: number;
}

interface RiskLimit {
  id: string;
  name: string;
  type: 'percentage' | 'absolute' | 'ratio';
  limit: number;
  current: number;
  status: 'safe' | 'warning' | 'danger';
  description: string;
}

interface RiskAlert {
  id: string;
  type: 'limit_breach' | 'high_volatility' | 'correlation_warning' | 'margin_call';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  symbol?: string;
  value?: number;
  acknowledged: boolean;
}

interface StressTestScenario {
  id: string;
  name: string;
  description: string;
  marketShock: number;
  volatilityIncrease: number;
  correlationIncrease: number;
  projectedLoss: number;
  probability: number;
}

interface HedgingPosition {
  id: string;
  symbol: string;
  type: 'hedge' | 'insurance';
  position: number;
  currentValue: number;
  hedgeRatio: number;
  effectiveness: number;
  cost: number;
  expiry?: string;
}

export default function RiskManagementPage() {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [showLimitModal, setShowLimitModal] = useState(false);
  const [showStressTestModal, setShowStressTestModal] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState<StressTestScenario | null>(null);

  // Fetch risk metrics
  const { data: riskMetrics, isLoading: metricsLoading } = useQuery<RiskMetrics>({
    queryKey: ['risk-metrics', selectedTimeframe],
    queryFn: async () => {
      const response = await api.get(`/risk/metrics?timeframe=${selectedTimeframe}`);
      return response.data;
    },
    refetchInterval: 30000, // 30 seconds
  });

  // Fetch risk limits
  const { data: riskLimits, isLoading: limitsLoading } = useQuery<RiskLimit[]>({
    queryKey: ['risk-limits'],
    queryFn: async () => {
      const response = await api.get('/risk/limits');
      return response.data;
    },
    refetchInterval: 10000, // 10 seconds
  });

  // Fetch risk alerts
  const { data: riskAlerts, isLoading: alertsLoading } = useQuery<RiskAlert[]>({
    queryKey: ['risk-alerts'],
    queryFn: async () => {
      const response = await api.get('/risk/alerts');
      return response.data;
    },
    refetchInterval: 5000, // 5 seconds
  });

  // Fetch stress test scenarios
  const { data: stressTestScenarios, isLoading: scenariosLoading } = useQuery<StressTestScenario[]>({
    queryKey: ['stress-test-scenarios'],
    queryFn: async () => {
      const response = await api.get('/risk/stress-tests');
      return response.data;
    },
  });

  // Fetch hedging positions
  const { data: hedgingPositions, isLoading: hedgingLoading } = useQuery<HedgingPosition[]>({
    queryKey: ['hedging-positions'],
    queryFn: async () => {
      const response = await api.get('/risk/hedging');
      return response.data;
    },
    refetchInterval: 15000, // 15 seconds
  });

  const timeframes = [
    { value: '1H', label: '1 Hour' },
    { value: '1D', label: '1 Day' },
    { value: '1W', label: '1 Week' },
    { value: '1M', label: '1 Month' },
  ];

  const getRiskScoreColor = (score: number) => {
    if (score <= 20) return 'text-green-600 dark:text-green-400';
    if (score <= 40) return 'text-yellow-600 dark:text-yellow-400';
    if (score <= 60) return 'text-orange-600 dark:text-orange-400';
    if (score <= 80) return 'text-red-600 dark:text-red-400';
    return 'text-red-700 dark:text-red-300';
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'safe': return 'success';
      case 'warning': return 'warning';
      case 'danger': return 'error';
      default: return 'info';
    }
  };

  const getSeverityBadgeVariant = (severity: string) => {
    switch (severity) {
      case 'low': return 'info';
      case 'medium': return 'warning';
      case 'high': return 'error';
      case 'critical': return 'error';
      default: return 'info';
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await api.post(`/risk/alerts/${alertId}/acknowledge`);
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const runStressTest = async (scenarioId: string) => {
    try {
      const response = await api.post(`/risk/stress-tests/${scenarioId}/run`);
      return response.data;
    } catch (error) {
      console.error('Failed to run stress test:', error);
    }
  };

  if (metricsLoading || limitsLoading || alertsLoading) {
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
            Risk Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Monitor and manage portfolio risk exposure
          </p>
        </div>
        <div className="flex space-x-3">
          <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
            {timeframes.map((timeframe) => (
              <button
                key={timeframe.value}
                onClick={() => setSelectedTimeframe(timeframe.value)}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  selectedTimeframe === timeframe.value
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                {timeframe.label}
              </button>
            ))}
          </div>
          <Button onClick={() => setShowStressTestModal(true)}>
            Run Stress Test
          </Button>
        </div>
      </div>

      {/* Risk Score Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="p-6 text-center">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Overall Risk Score
          </div>
          <div className={`text-4xl font-bold ${getRiskScoreColor(riskMetrics?.riskScore || 0)}`}>
            {riskMetrics?.riskScore || 0}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
            / 100
          </div>
        </Card>

        <Card className="p-6">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Portfolio Value
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatCurrency(riskMetrics?.portfolioValue || 0)}
          </div>
          <div className="text-xs text-green-600 mt-1">
            +2.5% today
          </div>
        </Card>

        <Card className="p-6">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Max Drawdown ({selectedTimeframe})
          </div>
          <div className="text-2xl font-bold text-red-600">
            {formatPercentage(riskMetrics?.maxDrawdown || 0)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
            Historical worst case
          </div>
        </Card>

        <Card className="p-6">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Margin Utilization
          </div>
          <div className="text-2xl font-bold text-orange-600">
            {formatPercentage(riskMetrics?.marginUtilization || 0)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
            of available margin
          </div>        </Card>
      </div>

      {/* Risk Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Score Gauge */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Risk Score Gauge
          </h3>
          <div className="h-64">
            <RiskGauge
              riskScore={riskMetrics?.riskScore || 0}
              maxRisk={100}
              thresholds={{
                low: 30,
                medium: 60,
                high: 80,
              }}
            />
          </div>
        </Card>

        {/* Risk Distribution */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Risk Distribution
          </h3>
          <div className="h-64">
            <DoughnutChart
              data={{
                labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
                datasets: [
                  {
                    data: [35, 30, 25, 10],
                    backgroundColor: [
                      'rgba(34, 197, 94, 0.8)',
                      'rgba(251, 191, 36, 0.8)',
                      'rgba(239, 68, 68, 0.8)',
                      'rgba(127, 29, 29, 0.8)',
                    ],
                    borderColor: [
                      'rgb(34, 197, 94)',
                      'rgb(251, 191, 36)',
                      'rgb(239, 68, 68)',
                      'rgb(127, 29, 29)',
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
                    position: 'bottom',
                  },
                  title: {
                    display: true,
                    text: 'Portfolio Risk Allocation',
                  },
                },
              }}
            />
          </div>
        </Card>
      </div>

      {/* Risk Metrics Details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Metrics */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Risk Metrics
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">VaR (95%)</span>
              <span className="font-medium text-red-600">
                {formatCurrency(riskMetrics?.var95 || 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">VaR (99%)</span>
              <span className="font-medium text-red-600">
                {formatCurrency(riskMetrics?.var99 || 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Expected Shortfall</span>
              <span className="font-medium text-red-600">
                {formatCurrency(riskMetrics?.expectedShortfall || 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</span>
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {(riskMetrics?.sharpeRatio || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Sortino Ratio</span>
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {(riskMetrics?.sortinoRatio || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Volatility</span>
              <span className="font-medium text-orange-600">
                {formatPercentage(riskMetrics?.volatility || 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Beta</span>
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {(riskMetrics?.beta || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Leverage Ratio</span>
              <span className="font-medium text-blue-600">
                {(riskMetrics?.leverageRatio || 0).toFixed(1)}x
              </span>
            </div>
          </div>
        </Card>

        {/* Risk Limits */}
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Risk Limits
            </h3>
            <Button variant="outline" size="sm" onClick={() => setShowLimitModal(true)}>
              Configure
            </Button>
          </div>
          {limitsLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-3">
              {riskLimits?.map((limit) => (
                <div key={limit.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {limit.name}
                      </span>
                      <Badge variant={getStatusBadgeVariant(limit.status)}>
                        {limit.status}
                      </Badge>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                      {limit.description}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {limit.type === 'percentage' 
                        ? formatPercentage(limit.current)
                        : limit.type === 'absolute'
                        ? formatCurrency(limit.current)
                        : limit.current.toFixed(2)
                      }
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500">
                      / {limit.type === 'percentage' 
                        ? formatPercentage(limit.limit)
                        : limit.type === 'absolute'
                        ? formatCurrency(limit.limit)
                        : limit.limit.toFixed(2)
                      }
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Risk Alerts */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Active Risk Alerts
        </h3>
        {alertsLoading ? (
          <Loading size="sm" />
        ) : riskAlerts && riskAlerts.length > 0 ? (
          <div className="space-y-3">
            {riskAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border-l-4 ${
                  alert.severity === 'critical'
                    ? 'bg-red-50 dark:bg-red-900/20 border-red-500'
                    : alert.severity === 'high'
                    ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-500'
                    : alert.severity === 'medium'
                    ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500'
                    : 'bg-blue-50 dark:bg-blue-900/20 border-blue-500'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <Badge variant={getSeverityBadgeVariant(alert.severity)}>
                        {alert.severity.toUpperCase()}
                      </Badge>
                      <span className="text-xs text-gray-500 dark:text-gray-500">
                        {alert.type.replace('_', ' ').toUpperCase()}
                      </span>
                      {alert.symbol && (
                        <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                          {alert.symbol}
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-900 dark:text-gray-100 mb-1">
                      {alert.message}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500">
                      {formatDate(alert.timestamp)}
                    </div>
                  </div>
                  {!alert.acknowledged && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => acknowledgeAlert(alert.id)}
                    >
                      Acknowledge
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500 dark:text-gray-500">
            No active risk alerts
          </div>
        )}
      </Card>

      {/* Stress Testing & Hedging */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Stress Test Scenarios */}
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Stress Test Scenarios
            </h3>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowStressTestModal(true)}
            >
              Run Tests
            </Button>
          </div>
          {scenariosLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-3">
              {stressTestScenarios?.map((scenario) => (
                <div
                  key={scenario.id}
                  className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                  onClick={() => setSelectedScenario(scenario)}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {scenario.name}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        {scenario.description}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-red-600">
                        {formatCurrency(scenario.projectedLoss)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-500">
                        {formatPercentage(scenario.probability)} probability
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Hedging Positions */}
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Hedging Positions
            </h3>
            <Button variant="outline" size="sm">
              Add Hedge
            </Button>
          </div>
          {hedgingLoading ? (
            <Loading size="sm" />
          ) : hedgingPositions && hedgingPositions.length > 0 ? (
            <div className="space-y-3">
              {hedgingPositions.map((hedge) => (
                <div key={hedge.id} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {hedge.symbol}
                        </span>
                        <Badge variant={hedge.type === 'hedge' ? 'info' : 'warning'}>
                          {hedge.type}
                        </Badge>
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        Effectiveness: {formatPercentage(hedge.effectiveness)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {formatCurrency(hedge.currentValue)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-500">
                        Cost: {formatCurrency(hedge.cost)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500 dark:text-gray-500">
              No hedging positions
            </div>
          )}
        </Card>
      </div>

      {/* Modals would go here - simplified for brevity */}
      {showStressTestModal && (
        <Modal
          isOpen={showStressTestModal}
          onClose={() => setShowStressTestModal(false)}
          title="Run Stress Tests"
          size="lg"
        >
          <div className="p-4">
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Select scenarios to test portfolio resilience under extreme market conditions.
            </p>
            {/* Stress test content would go here */}
            <div className="flex justify-end space-x-3 mt-6">
              <Button variant="outline" onClick={() => setShowStressTestModal(false)}>
                Cancel
              </Button>
              <Button onClick={() => setShowStressTestModal(false)}>
                Run Selected Tests
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
