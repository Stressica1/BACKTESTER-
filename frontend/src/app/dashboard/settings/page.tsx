'use client';

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Loading } from '@/components/ui/Loading';
import { Input } from '@/components/ui/Input';
import { api } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

interface UserSettings {
  notifications: {
    email: boolean;
    push: boolean;
    trading_alerts: boolean;
    portfolio_updates: boolean;
    news_updates: boolean;
    system_alerts: boolean;
  };
  display: {
    theme: 'light' | 'dark' | 'auto';
    currency: string;
    timezone: string;
    number_format: 'US' | 'EU' | 'UK';
    chart_type: 'candlestick' | 'line' | 'area';
  };
  trading: {
    default_leverage: number;
    max_risk_per_trade: number;
    auto_stop_loss: boolean;
    default_stop_loss: number;
    default_take_profit: number;
    position_sizing_method: 'fixed' | 'percentage' | 'risk_based';
  };
  api: {
    binance_enabled: boolean;
    binance_api_key: string;
    coinbase_enabled: boolean;
    coinbase_api_key: string;
    ftx_enabled: boolean;
    ftx_api_key: string;
    webhook_url: string;
    rate_limit: number;
  };
  security: {
    two_factor_enabled: boolean;
    session_timeout: number;
    ip_whitelist: string[];
    login_notifications: boolean;
  };
}

interface SystemInfo {
  version: string;
  uptime: string;
  last_backup: string;
  storage_used: number;
  storage_total: number;
  active_connections: number;
  cpu_usage: number;
  memory_usage: number;
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<'general' | 'trading' | 'api' | 'security' | 'system'>('general');
  const [showApiKeys, setShowApiKeys] = useState(false);
  const { user } = useAuthStore();
  const queryClient = useQueryClient();

  // Fetch user settings
  const { data: settings, isLoading: settingsLoading } = useQuery<UserSettings>({
    queryKey: ['user-settings'],
    queryFn: async () => {
      const response = await api.get('/user/settings');
      return response.data;
    },
  });

  // Fetch system info
  const { data: systemInfo, isLoading: systemLoading } = useQuery<SystemInfo>({
    queryKey: ['system-info'],
    queryFn: async () => {
      const response = await api.get('/system/info');
      return response.data;
    },
    refetchInterval: 30000, // 30 seconds
  });

  // Update settings mutation
  const updateSettingsMutation = useMutation({
    mutationFn: async (updates: Partial<UserSettings>) => {
      const response = await api.put('/user/settings', updates);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user-settings'] });
    },
  });

  // Test API connection mutation
  const testApiMutation = useMutation({
    mutationFn: async (exchange: string) => {
      const response = await api.post(`/api/test-connection/${exchange}`);
      return response.data;
    },
  });

  const tabs = [
    { id: 'general', label: 'General', icon: '⚙️' },
    { id: 'trading', label: 'Trading', icon: '📈' },
    { id: 'api', label: 'API & Exchanges', icon: '🔗' },
    { id: 'security', label: 'Security', icon: '🔒' },
    { id: 'system', label: 'System', icon: '💻' },
  ];

  const currencies = [
    { value: 'USD', label: 'US Dollar ($)' },
    { value: 'EUR', label: 'Euro (€)' },
    { value: 'GBP', label: 'British Pound (£)' },
    { value: 'JPY', label: 'Japanese Yen (¥)' },
    { value: 'BTC', label: 'Bitcoin (₿)' },
  ];

  const timezones = [
    'UTC',
    'America/New_York',
    'America/Los_Angeles',
    'Europe/London',
    'Europe/Berlin',
    'Asia/Tokyo',
    'Asia/Shanghai',
    'Australia/Sydney',
  ];

  const handleSettingChange = (section: keyof UserSettings, key: string, value: any) => {
    if (!settings) return;
    
    const updatedSettings = {
      ...settings,
      [section]: {
        ...settings[section],
        [key]: value,
      },
    };
    
    updateSettingsMutation.mutate({ [section]: updatedSettings[section] });
  };

  if (settingsLoading) {
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
            Settings
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Manage your account preferences and system configuration
          </p>
        </div>
        <Button onClick={() => updateSettingsMutation.mutate(settings!)}>
          Save All Changes
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Settings Navigation */}
        <Card className="p-4 h-fit">
          <nav className="space-y-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                <span>{tab.icon}</span>
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </nav>
        </Card>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          {activeTab === 'general' && (
            <div className="space-y-6">
              {/* Profile Information */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Profile Information
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Full Name
                    </label>
                    <Input value={user?.name || ''} />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Email
                    </label>
                    <Input value={user?.email || ''} disabled />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Phone Number
                    </label>
                    <Input placeholder="+1 (555) 123-4567" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Country
                    </label>
                    <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md">
                      <option>United States</option>
                      <option>United Kingdom</option>
                      <option>Canada</option>
                      <option>Germany</option>
                      <option>Japan</option>
                    </select>
                  </div>
                </div>
              </Card>

              {/* Display Settings */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Display Settings
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Theme
                    </label>
                    <select
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                      value={settings?.display.theme || 'auto'}
                      onChange={(e) => handleSettingChange('display', 'theme', e.target.value)}
                    >
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                      <option value="auto">Auto (System)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Currency
                    </label>
                    <select
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                      value={settings?.display.currency || 'USD'}
                      onChange={(e) => handleSettingChange('display', 'currency', e.target.value)}
                    >
                      {currencies.map((currency) => (
                        <option key={currency.value} value={currency.value}>
                          {currency.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Timezone
                    </label>
                    <select
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                      value={settings?.display.timezone || 'UTC'}
                      onChange={(e) => handleSettingChange('display', 'timezone', e.target.value)}
                    >
                      {timezones.map((tz) => (
                        <option key={tz} value={tz}>
                          {tz}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Chart Type
                    </label>
                    <select
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                      value={settings?.display.chart_type || 'candlestick'}
                      onChange={(e) => handleSettingChange('display', 'chart_type', e.target.value)}
                    >
                      <option value="candlestick">Candlestick</option>
                      <option value="line">Line</option>
                      <option value="area">Area</option>
                    </select>
                  </div>
                </div>
              </Card>

              {/* Notification Settings */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Notifications
                </h3>
                <div className="space-y-4">
                  {[
                    { key: 'email', label: 'Email Notifications' },
                    { key: 'push', label: 'Push Notifications' },
                    { key: 'trading_alerts', label: 'Trading Alerts' },
                    { key: 'portfolio_updates', label: 'Portfolio Updates' },
                    { key: 'news_updates', label: 'News & Market Updates' },
                    { key: 'system_alerts', label: 'System Alerts' },
                  ].map((notification) => (
                    <div key={notification.key} className="flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {notification.label}
                        </div>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          className="sr-only peer"
                          checked={settings?.notifications[notification.key as keyof typeof settings.notifications] || false}
                          onChange={(e) => handleSettingChange('notifications', notification.key, e.target.checked)}
                        />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                      </label>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          )}

          {activeTab === 'trading' && (
            <div className="space-y-6">
              {/* Default Trading Settings */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Default Trading Parameters
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Default Leverage
                    </label>
                    <Input
                      type="number"
                      min="1"
                      max="100"
                      value={settings?.trading.default_leverage || 1}
                      onChange={(e) => handleSettingChange('trading', 'default_leverage', Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Max Risk per Trade (%)
                    </label>
                    <Input
                      type="number"
                      min="0.1"
                      max="10"
                      step="0.1"
                      value={settings?.trading.max_risk_per_trade || 2}
                      onChange={(e) => handleSettingChange('trading', 'max_risk_per_trade', Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Default Stop Loss (%)
                    </label>
                    <Input
                      type="number"
                      min="0.1"
                      max="50"
                      step="0.1"
                      value={settings?.trading.default_stop_loss || 2}
                      onChange={(e) => handleSettingChange('trading', 'default_stop_loss', Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Default Take Profit (%)
                    </label>
                    <Input
                      type="number"
                      min="0.1"
                      max="100"
                      step="0.1"
                      value={settings?.trading.default_take_profit || 4}
                      onChange={(e) => handleSettingChange('trading', 'default_take_profit', Number(e.target.value))}
                    />
                  </div>
                </div>
              </Card>

              {/* Position Sizing */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Position Sizing
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Position Sizing Method
                    </label>
                    <select
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                      value={settings?.trading.position_sizing_method || 'percentage'}
                      onChange={(e) => handleSettingChange('trading', 'position_sizing_method', e.target.value)}
                    >
                      <option value="fixed">Fixed Amount</option>
                      <option value="percentage">Percentage of Portfolio</option>
                      <option value="risk_based">Risk-Based Sizing</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        Auto Stop Loss
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-500">
                        Automatically set stop loss on new positions
                      </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        className="sr-only peer"
                        checked={settings?.trading.auto_stop_loss || false}
                        onChange={(e) => handleSettingChange('trading', 'auto_stop_loss', e.target.checked)}
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                    </label>
                  </div>
                </div>
              </Card>
            </div>
          )}

          {activeTab === 'api' && (
            <div className="space-y-6">
              {/* Exchange API Settings */}
              <Card className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    Exchange API Configuration
                  </h3>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowApiKeys(!showApiKeys)}
                  >
                    {showApiKeys ? 'Hide' : 'Show'} API Keys
                  </Button>
                </div>

                <div className="space-y-6">
                  {/* Binance */}
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center text-white font-bold">
                          B
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-gray-100">Binance</h4>
                          <p className="text-sm text-gray-500 dark:text-gray-500">
                            Connect your Binance account for live trading
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <Badge variant={settings?.api.binance_enabled ? 'success' : 'default'}>
                          {settings?.api.binance_enabled ? 'Connected' : 'Disconnected'}
                        </Badge>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => testApiMutation.mutate('binance')}
                        >
                          Test
                        </Button>
                      </div>
                    </div>
                    
                    {showApiKeys && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            API Key
                          </label>
                          <Input
                            type="password"
                            placeholder="Enter Binance API Key"
                            value={settings?.api.binance_api_key || ''}
                            onChange={(e) => handleSettingChange('api', 'binance_api_key', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            Secret Key
                          </label>
                          <Input
                            type="password"
                            placeholder="Enter Secret Key"
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Coinbase */}
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold">
                          C
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-gray-100">Coinbase Pro</h4>
                          <p className="text-sm text-gray-500 dark:text-gray-500">
                            Connect your Coinbase Pro account
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <Badge variant={settings?.api.coinbase_enabled ? 'success' : 'default'}>
                          {settings?.api.coinbase_enabled ? 'Connected' : 'Disconnected'}
                        </Badge>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => testApiMutation.mutate('coinbase')}
                        >
                          Test
                        </Button>
                      </div>
                    </div>
                    
                    {showApiKeys && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            API Key
                          </label>
                          <Input
                            type="password"
                            placeholder="Enter Coinbase API Key"
                            value={settings?.api.coinbase_api_key || ''}
                            onChange={(e) => handleSettingChange('api', 'coinbase_api_key', e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            Secret Key
                          </label>
                          <Input
                            type="password"
                            placeholder="Enter Secret Key"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </Card>

              {/* Webhook Settings */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Webhook Configuration
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Webhook URL
                    </label>
                    <Input
                      placeholder="https://your-domain.com/webhook"
                      value={settings?.api.webhook_url || ''}
                      onChange={(e) => handleSettingChange('api', 'webhook_url', e.target.value)}
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                      URL to receive TradingView alerts and other webhook notifications
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Rate Limit (requests/minute)
                    </label>
                    <Input
                      type="number"
                      min="1"
                      max="1000"
                      value={settings?.api.rate_limit || 60}
                      onChange={(e) => handleSettingChange('api', 'rate_limit', Number(e.target.value))}
                    />
                  </div>
                </div>
              </Card>
            </div>
          )}

          {activeTab === 'security' && (
            <div className="space-y-6">
              {/* Two-Factor Authentication */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Two-Factor Authentication
                </h3>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      Enable 2FA
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500">
                      Add an extra layer of security to your account
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Badge variant={settings?.security.two_factor_enabled ? 'success' : 'warning'}>
                      {settings?.security.two_factor_enabled ? 'Enabled' : 'Disabled'}
                    </Badge>
                    <Button variant="outline" size="sm">
                      {settings?.security.two_factor_enabled ? 'Disable' : 'Setup'}
                    </Button>
                  </div>
                </div>
              </Card>

              {/* Session Settings */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Session Security
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Session Timeout (minutes)
                    </label>
                    <select
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md"
                      value={settings?.security.session_timeout || 60}
                      onChange={(e) => handleSettingChange('security', 'session_timeout', Number(e.target.value))}
                    >
                      <option value={15}>15 minutes</option>
                      <option value={30}>30 minutes</option>
                      <option value={60}>1 hour</option>
                      <option value={240}>4 hours</option>
                      <option value={480}>8 hours</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        Login Notifications
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-500">
                        Get notified when someone logs into your account
                      </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        className="sr-only peer"
                        checked={settings?.security.login_notifications || false}
                        onChange={(e) => handleSettingChange('security', 'login_notifications', e.target.checked)}
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                    </label>
                  </div>
                </div>
              </Card>

              {/* IP Whitelist */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  IP Whitelist
                </h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Add IP Address
                    </label>
                    <div className="flex space-x-2">
                      <Input placeholder="192.168.1.1" className="flex-1" />
                      <Button size="sm">Add</Button>
                    </div>
                  </div>
                  
                  {settings?.security.ip_whitelist && settings.security.ip_whitelist.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Whitelisted IPs
                      </div>
                      {settings.security.ip_whitelist.map((ip, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded">
                          <span className="text-sm text-gray-900 dark:text-gray-100">{ip}</span>
                          <Button variant="outline" size="sm">Remove</Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}

          {activeTab === 'system' && (
            <div className="space-y-6">
              {/* System Information */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  System Information
                </h3>
                {systemLoading ? (
                  <Loading size="sm" />
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Version</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {systemInfo?.version || 'v1.0.0'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Uptime</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {systemInfo?.uptime || '24h 15m'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Last Backup</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {systemInfo?.last_backup || '2 hours ago'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Active Connections</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {systemInfo?.active_connections || 5}
                        </span>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">CPU Usage</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {systemInfo?.cpu_usage || 25}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Memory Usage</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {systemInfo?.memory_usage || 45}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Storage Used</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {systemInfo?.storage_used || 2.1}GB / {systemInfo?.storage_total || 10}GB
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </Card>

              {/* System Actions */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  System Actions
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button variant="outline" className="justify-start">
                    🔄 Restart Services
                  </Button>
                  <Button variant="outline" className="justify-start">
                    💾 Create Backup
                  </Button>
                  <Button variant="outline" className="justify-start">
                    📊 Export Data
                  </Button>
                  <Button variant="outline" className="justify-start">
                    🧹 Clear Cache
                  </Button>
                  <Button variant="outline" className="justify-start">
                    📋 View Logs
                  </Button>
                  <Button variant="outline" className="justify-start">
                    🔍 Run Diagnostics
                  </Button>
                </div>
              </Card>

              {/* Danger Zone */}
              <Card className="p-6 border-red-200 dark:border-red-800">
                <h3 className="text-lg font-semibold text-red-700 dark:text-red-400 mb-4">
                  Danger Zone
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                    <div>
                      <div className="text-sm font-medium text-red-700 dark:text-red-400">
                        Reset All Settings
                      </div>
                      <div className="text-xs text-red-600 dark:text-red-500">
                        This will reset all your settings to default values
                      </div>
                    </div>
                    <Button variant="error" size="sm">
                      Reset
                    </Button>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                    <div>
                      <div className="text-sm font-medium text-red-700 dark:text-red-400">
                        Delete Account
                      </div>
                      <div className="text-xs text-red-600 dark:text-red-500">
                        Permanently delete your account and all data
                      </div>
                    </div>
                    <Button variant="error" size="sm">
                      Delete
                    </Button>
                  </div>
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
