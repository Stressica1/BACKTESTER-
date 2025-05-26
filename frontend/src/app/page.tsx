'use client';

import React from 'react';
import { redirect } from 'next/navigation';
import { useAuthStore } from '@/lib/store';
import { Button } from '@/components/ui/Button';
import { ArrowRightIcon, ChartBarIcon, ShieldCheckIcon, CpuChipIcon, BoltIcon } from '@heroicons/react/24/outline';

export default function HomePage() {
  const { isAuthenticated } = useAuthStore();

  if (isAuthenticated) {
    redirect('/dashboard');
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation */}
      <nav className="relative z-10 flex items-center justify-between p-6 lg:px-8">
        <div className="flex items-center space-x-2">
          <ChartBarIcon className="h-8 w-8 text-blue-400" />
          <span className="text-xl font-bold text-white">QuantTrade Pro</span>
        </div>
        <div className="space-x-4">
          <Button variant="ghost" size="sm" href="/login">
            Sign In
          </Button>
          <Button variant="primary" size="sm" href="/register">
            Get Started
          </Button>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative px-6 lg:px-8">
        <div className="mx-auto max-w-4xl pt-20 pb-32 sm:pt-32 sm:pb-40">
          <div className="text-center">
            <h1 className="text-4xl font-bold tracking-tight text-white sm:text-6xl">
              Enterprise-Grade
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {' '}Quantitative Trading
              </span>
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-300 max-w-2xl mx-auto">
              Advanced AI-powered trading platform with real-time analytics, risk management, 
              and institutional-grade infrastructure. Trade smarter with data-driven strategies.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Button 
                variant="primary" 
                size="lg" 
                href="/register"
                className="flex items-center space-x-2"
              >
                <span>Start Trading</span>
                <ArrowRightIcon className="h-4 w-4" />
              </Button>
              <Button 
                variant="ghost" 
                size="lg" 
                href="#features"
                className="text-white border-white/20 hover:bg-white/10"
              >
                Learn More
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="bg-white dark:bg-gray-900 py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-4xl">
              Powerful Trading Features
            </h2>
            <p className="mt-6 text-lg leading-8 text-gray-600 dark:text-gray-300">
              Everything you need to build, test, and deploy successful trading strategies.
            </p>
          </div>
          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-3">
              <div className="flex flex-col">
                <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 dark:text-white">
                  <CpuChipIcon className="h-5 w-5 flex-none text-blue-600" aria-hidden="true" />
                  AI-Powered Strategies
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600 dark:text-gray-300">
                  <p className="flex-auto">
                    Advanced machine learning algorithms including LSTM networks, reinforcement learning, 
                    and ensemble methods for optimal strategy development and execution.
                  </p>
                </dd>
              </div>
              <div className="flex flex-col">
                <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 dark:text-white">
                  <ShieldCheckIcon className="h-5 w-5 flex-none text-blue-600" aria-hidden="true" />
                  Advanced Risk Management
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600 dark:text-gray-300">
                  <p className="flex-auto">
                    Comprehensive risk controls with VaR calculations, stress testing, real-time monitoring, 
                    and automated position management to protect your capital.
                  </p>
                </dd>
              </div>
              <div className="flex flex-col">
                <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900 dark:text-white">
                  <BoltIcon className="h-5 w-5 flex-none text-blue-600" aria-hidden="true" />
                  Real-Time Analytics
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600 dark:text-gray-300">
                  <p className="flex-auto">
                    Live market data feeds, interactive charts, performance metrics, and customizable 
                    dashboards with institutional-grade analytics and reporting.
                  </p>
                </dd>
              </div>
            </dl>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-gray-50 dark:bg-gray-800 py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <dl className="grid grid-cols-1 gap-x-8 gap-y-16 text-center lg:grid-cols-3">
            <div className="mx-auto flex max-w-xs flex-col gap-y-4">
              <dt className="text-base leading-7 text-gray-600 dark:text-gray-300">
                Active Strategies
              </dt>
              <dd className="order-first text-3xl font-semibold tracking-tight text-gray-900 dark:text-white sm:text-5xl">
                1000+
              </dd>
            </div>
            <div className="mx-auto flex max-w-xs flex-col gap-y-4">
              <dt className="text-base leading-7 text-gray-600 dark:text-gray-300">
                Assets Under Management
              </dt>
              <dd className="order-first text-3xl font-semibold tracking-tight text-gray-900 dark:text-white sm:text-5xl">
                $500M+
              </dd>
            </div>
            <div className="mx-auto flex max-w-xs flex-col gap-y-4">
              <dt className="text-base leading-7 text-gray-600 dark:text-gray-300">
                Average Daily Volume
              </dt>
              <dd className="order-first text-3xl font-semibold tracking-tight text-gray-900 dark:text-white sm:text-5xl">
                $50M+
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-blue-600">
        <div className="px-6 py-24 sm:px-6 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Ready to start trading?
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-blue-200">
              Join thousands of traders using our platform to build and execute profitable strategies.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Button 
                variant="secondary" 
                size="lg"
                href="/register"
                className="bg-white text-blue-600 hover:bg-gray-50"
              >
                Get Started Free
              </Button>
              <Button 
                variant="ghost" 
                size="lg"
                href="/login"
                className="text-white border-white/20 hover:bg-white/10"
              >
                Sign In
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900">
        <div className="mx-auto max-w-7xl px-6 py-12 md:flex md:items-center md:justify-between lg:px-8">
          <div className="flex justify-center space-x-6 md:order-2">
            <p className="text-center text-xs leading-5 text-gray-400">
              &copy; 2024 QuantTrade Pro. All rights reserved.
            </p>
          </div>
          <div className="mt-8 md:order-1 md:mt-0">
            <div className="flex items-center space-x-2">
              <ChartBarIcon className="h-6 w-6 text-blue-400" />
              <span className="text-sm font-bold text-white">QuantTrade Pro</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
