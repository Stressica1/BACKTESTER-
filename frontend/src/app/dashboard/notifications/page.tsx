'use client';

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  BellIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  InfoIcon,
  SettingsIcon,
  FilterIcon,
  MarkAsReadIcon,
  Trash2Icon,
  RefreshCwIcon,
  SearchIcon,
} from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Loading } from '@/components/ui/Loading';
import { Badge } from '@/components/ui/Badge';
import { Modal, ModalHeader, ModalBody, ModalFooter } from '@/components/ui/Modal';
import { api } from '@/lib/api';
import { formatCurrency, formatNumber } from '@/lib/utils';

interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success' | 'trade' | 'system' | 'price_alert';
  title: string;
  message: string;
  data?: any;
  isRead: boolean;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  timestamp: string;
  source: string;
  actionUrl?: string;
}

interface NotificationSettings {
  emailNotifications: boolean;
  pushNotifications: boolean;
  tradeAlerts: boolean;
  priceAlerts: boolean;
  systemAlerts: boolean;
  portfolioAlerts: boolean;
  riskAlerts: boolean;
  minimumAlertAmount: number;
}

export default function NotificationsPage() {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<'all' | 'unread' | 'trade' | 'system' | 'price_alert'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [selectedNotification, setSelectedNotification] = useState<Notification | null>(null);

  // Fetch notifications
  const { data: notifications = [], isLoading: notificationsLoading, refetch } = useQuery<Notification[]>({
    queryKey: ['notifications', filter, searchQuery],
    queryFn: () => api.get(`/notifications?filter=${filter}&search=${searchQuery}`),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch notification settings
  const { data: settings, isLoading: settingsLoading } = useQuery<NotificationSettings>({
    queryKey: ['notification-settings'],
    queryFn: () => api.get('/notifications/settings'),
  });

  // Mark as read mutation
  const markAsReadMutation = useMutation({
    mutationFn: (notificationId: string) => api.patch(`/notifications/${notificationId}/read`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
    },
  });

  // Mark all as read mutation
  const markAllAsReadMutation = useMutation({
    mutationFn: () => api.patch('/notifications/mark-all-read'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
    },
  });

  // Delete notification mutation
  const deleteNotificationMutation = useMutation({
    mutationFn: (notificationId: string) => api.delete(`/notifications/${notificationId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
    },
  });

  // Update settings mutation
  const updateSettingsMutation = useMutation({
    mutationFn: (newSettings: Partial<NotificationSettings>) => 
      api.patch('/notifications/settings', newSettings),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notification-settings'] });
      setShowSettings(false);
    },
  });

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <AlertTriangleIcon className="w-5 h-5 text-yellow-500" />;
      case 'error':
        return <XCircleIcon className="w-5 h-5 text-red-500" />;
      case 'trade':
        return <BellIcon className="w-5 h-5 text-blue-500" />;
      case 'price_alert':
        return <BellIcon className="w-5 h-5 text-purple-500" />;
      default:
        return <InfoIcon className="w-5 h-5 text-blue-500" />;
    }
  };

  const getPriorityBadge = (priority: string) => {
    const priorityConfig = {
      low: { variant: 'secondary' as const, text: 'Low' },
      medium: { variant: 'info' as const, text: 'Medium' },
      high: { variant: 'warning' as const, text: 'High' },
      urgent: { variant: 'error' as const, text: 'Urgent' },
    };
    
    const config = priorityConfig[priority as keyof typeof priorityConfig] || priorityConfig.low;
    return <Badge variant={config.variant} size="sm">{config.text}</Badge>;
  };

  const unreadCount = notifications.filter(n => !n.isRead).length;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 flex items-center">
            <BellIcon className="w-8 h-8 mr-3" />
            Notifications
            {unreadCount > 0 && (
              <Badge variant="error" className="ml-3">
                {unreadCount} unread
              </Badge>
            )}
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            Stay updated with trading alerts and system notifications
          </p>
        </div>
        <div className="mt-4 lg:mt-0 flex items-center space-x-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
          >
            <RefreshCwIcon className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          {unreadCount > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => markAllAsReadMutation.mutate()}
              disabled={markAllAsReadMutation.isPending}
            >
              <MarkAsReadIcon className="w-4 h-4 mr-2" />
              Mark All Read
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSettings(true)}
          >
            <SettingsIcon className="w-4 h-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
        <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
          {[
            { key: 'all', label: 'All' },
            { key: 'unread', label: 'Unread' },
            { key: 'trade', label: 'Trading' },
            { key: 'system', label: 'System' },
            { key: 'price_alert', label: 'Price Alerts' },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setFilter(key as any)}
              className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                filter === key
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        
        <div className="relative">
          <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <Input
            type="text"
            placeholder="Search notifications..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 w-64"
          />
        </div>
      </div>

      {/* Notifications List */}
      <Card className="p-6">
        {notificationsLoading ? (
          <Loading size="md" className="mx-auto" />
        ) : (
          <div className="space-y-4">
            {notifications.map((notification) => (
              <div
                key={notification.id}
                className={`p-4 rounded-lg border transition-colors cursor-pointer ${
                  notification.isRead
                    ? 'bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700'
                    : 'bg-blue-50 dark:bg-blue-900/10 border-blue-200 dark:border-blue-800'
                }`}
                onClick={() => setSelectedNotification(notification)}
              >
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    {getNotificationIcon(notification.type)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <h4 className={`text-sm font-medium ${
                          notification.isRead ? 'text-gray-900 dark:text-gray-100' : 'text-blue-900 dark:text-blue-100'
                        }`}>
                          {notification.title}
                        </h4>
                        {getPriorityBadge(notification.priority)}
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {new Date(notification.timestamp).toLocaleString()}
                        </span>
                        {!notification.isRead && (
                          <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                        )}
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 line-clamp-2">
                      {notification.message}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        Source: {notification.source}
                      </span>
                      
                      <div className="flex items-center space-x-2">
                        {!notification.isRead && (
                          <Button
                            variant="ghost"
                            size="xs"
                            onClick={(e) => {
                              e.stopPropagation();
                              markAsReadMutation.mutate(notification.id);
                            }}
                          >
                            <MarkAsReadIcon className="w-3 h-3" />
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="xs"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteNotificationMutation.mutate(notification.id);
                          }}
                        >
                          <Trash2Icon className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {notifications.length === 0 && (
              <div className="text-center py-12">
                <BellIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                  No notifications
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  {filter === 'all' 
                    ? "You're all caught up! No notifications to show."
                    : `No ${filter} notifications found.`}
                </p>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Notification Detail Modal */}
      {selectedNotification && (
        <Modal
          isOpen={!!selectedNotification}
          onClose={() => setSelectedNotification(null)}
          title="Notification Details"
          size="md"
        >
          <ModalBody>
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                {getNotificationIcon(selectedNotification.type)}
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-gray-100">
                    {selectedNotification.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {new Date(selectedNotification.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                {getPriorityBadge(selectedNotification.priority)}
                <Badge variant="secondary" size="sm">
                  {selectedNotification.source}
                </Badge>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                <p className="text-gray-900 dark:text-gray-100">
                  {selectedNotification.message}
                </p>
              </div>
              
              {selectedNotification.data && (
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
                    Additional Details
                  </h4>
                  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <pre className="text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                      {JSON.stringify(selectedNotification.data, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </ModalBody>
          
          <ModalFooter>
            <Button
              variant="outline"
              onClick={() => setSelectedNotification(null)}
            >
              Close
            </Button>
            {!selectedNotification.isRead && (
              <Button
                variant="primary"
                onClick={() => {
                  markAsReadMutation.mutate(selectedNotification.id);
                  setSelectedNotification(null);
                }}
              >
                Mark as Read
              </Button>
            )}
            {selectedNotification.actionUrl && (
              <Button
                variant="primary"
                onClick={() => {
                  window.open(selectedNotification.actionUrl, '_blank');
                }}
              >
                Take Action
              </Button>
            )}
          </ModalFooter>
        </Modal>
      )}

      {/* Settings Modal */}
      <Modal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        title="Notification Settings"
        size="md"
      >
        <ModalBody>
          {settingsLoading ? (
            <Loading size="sm" />
          ) : (
            <div className="space-y-6">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-4">
                  Notification Channels
                </h4>
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings?.emailNotifications || false}
                      onChange={(e) => {
                        if (settings) {
                          updateSettingsMutation.mutate({
                            emailNotifications: e.target.checked
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                      Email Notifications
                    </span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings?.pushNotifications || false}
                      onChange={(e) => {
                        if (settings) {
                          updateSettingsMutation.mutate({
                            pushNotifications: e.target.checked
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                      Push Notifications
                    </span>
                  </label>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-4">
                  Alert Types
                </h4>
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings?.tradeAlerts || false}
                      onChange={(e) => {
                        if (settings) {
                          updateSettingsMutation.mutate({
                            tradeAlerts: e.target.checked
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                      Trade Execution Alerts
                    </span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings?.priceAlerts || false}
                      onChange={(e) => {
                        if (settings) {
                          updateSettingsMutation.mutate({
                            priceAlerts: e.target.checked
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                      Price Alerts
                    </span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings?.portfolioAlerts || false}
                      onChange={(e) => {
                        if (settings) {
                          updateSettingsMutation.mutate({
                            portfolioAlerts: e.target.checked
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                      Portfolio Performance Alerts
                    </span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings?.riskAlerts || false}
                      onChange={(e) => {
                        if (settings) {
                          updateSettingsMutation.mutate({
                            riskAlerts: e.target.checked
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                      Risk Management Alerts
                    </span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings?.systemAlerts || false}
                      onChange={(e) => {
                        if (settings) {
                          updateSettingsMutation.mutate({
                            systemAlerts: e.target.checked
                          });
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                      System Alerts
                    </span>
                  </label>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Minimum Alert Amount ($)
                </label>
                <Input
                  type="number"
                  value={settings?.minimumAlertAmount || 0}
                  onChange={(e) => {
                    if (settings) {
                      updateSettingsMutation.mutate({
                        minimumAlertAmount: parseFloat(e.target.value) || 0
                      });
                    }
                  }}
                  min="0"
                  step="1"
                  placeholder="1000"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Only send alerts for trades above this amount
                </p>
              </div>
            </div>
          )}
        </ModalBody>
        
        <ModalFooter>
          <Button
            variant="outline"
            onClick={() => setShowSettings(false)}
          >
            Close
          </Button>
        </ModalFooter>
      </Modal>
    </div>
  );
}
