'use client';

import React from 'react';
import {
  Chart as ChartJS,
  RadialLinearScale,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { PolarArea } from 'react-chartjs-2';
import { chartColors } from '@/lib/chartConfig';
import { useThemeStore } from '@/lib/store';

ChartJS.register(RadialLinearScale, ArcElement, Tooltip, Legend);

interface RiskGaugeProps {
  value: number; // 0-100
  maxValue?: number;
  label?: string;
  height?: string;
  riskLevels?: {
    low: number;
    medium: number;
    high: number;
  };
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({
  value,
  maxValue = 100,
  label = 'Risk Score',
  height = '200px',
  riskLevels = { low: 30, medium: 70, high: 100 },
}) => {
  const { theme } = useThemeStore();

  // Determine risk level and color
  const getRiskLevel = (val: number) => {
    if (val <= riskLevels.low) return { level: 'Low', color: chartColors.success };
    if (val <= riskLevels.medium) return { level: 'Medium', color: chartColors.warning };
    return { level: 'High', color: chartColors.danger };
  };

  const risk = getRiskLevel(value);

  const data = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Remaining'],
    datasets: [
      {
        data: [
          Math.min(value, riskLevels.low),
          Math.max(0, Math.min(value - riskLevels.low, riskLevels.medium - riskLevels.low)),
          Math.max(0, value - riskLevels.medium),
          Math.max(0, maxValue - value),
        ],
        backgroundColor: [
          chartColors.success,
          chartColors.warning,
          chartColors.danger,
          theme === 'dark' ? '#374151' : '#E5E7EB',
        ],
        borderColor: [
          chartColors.success,
          chartColors.warning,
          chartColors.danger,
          theme === 'dark' ? '#374151' : '#E5E7EB',
        ],
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false,
      },
    },
    scales: {
      r: {
        angleLines: {
          display: false,
        },
        grid: {
          color: theme === 'dark' ? '#374151' : '#E5E7EB',
        },
        pointLabels: {
          display: false,
        },
        ticks: {
          display: false,
        },
        suggestedMin: 0,
        suggestedMax: maxValue,
      },
    },
    elements: {
      arc: {
        borderWidth: 0,
        borderRadius: 4,
      },
    },
  };

  return (
    <div style={{ height }} className="relative">
      <PolarArea data={data} options={options} />
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="text-center">
          <div className="text-3xl font-bold" style={{ color: risk.color }}>
            {value}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {label}
          </div>
          <div 
            className="text-xs font-medium mt-1"
            style={{ color: risk.color }}
          >
            {risk.level} Risk
          </div>
        </div>
      </div>
    </div>
  );
};
