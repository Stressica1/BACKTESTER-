'use client';

import React from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import dynamic from 'next/dynamic';
import { getDoughnutChartOptions, chartColors } from '@/lib/chartConfig';
import { useThemeStore } from '@/lib/store';

ChartJS.register(ArcElement, Tooltip, Legend);

// Dynamically load chart component to reduce bundle size and disable SSR
const Doughnut = dynamic(
  () => import('react-chartjs-2').then((mod) => mod.Doughnut),
  {
    ssr: false,
    loading: () => <div className="animate-pulse bg-gray-200 h-full w-full rounded-lg" />
  }
);

interface DoughnutChartProps {
  data: {
    labels: string[];
    datasets: {
      data: number[];
      backgroundColor?: string[];
      borderColor?: string[];
      borderWidth?: number;
    }[];
  };
  height?: string;
  showLegend?: boolean;
  centerText?: string;
  centerSubtext?: string;
}

export const DoughnutChart: React.FC<DoughnutChartProps> = ({
  data,
  height = '300px',
  showLegend = true,
  centerText,
  centerSubtext,
}) => {
  const { theme } = useThemeStore();
  const chartOptions = getDoughnutChartOptions(theme);

  // Generate colors if not provided
  const defaultColors = [
    chartColors.primary,
    chartColors.success,
    chartColors.warning,
    chartColors.danger,
    chartColors.info,
    chartColors.purple,
    chartColors.pink,
  ];

  const processedData = {
    ...data,
    datasets: data.datasets.map((dataset) => ({
      ...dataset,
      backgroundColor: dataset.backgroundColor || defaultColors,
      borderColor: dataset.borderColor || defaultColors,
      borderWidth: dataset.borderWidth || 2,
    })),
  };

  const options = {
    ...chartOptions,
    plugins: {
      ...chartOptions.plugins,
      legend: {
        ...chartOptions.plugins.legend,
        display: showLegend,
      },
    },
  };

  return (
    <div style={{ height }} className="relative will-change-transform">
      <Doughnut data={processedData} options={options} />
      {(centerText || centerSubtext) && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            {centerText && (
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {centerText}
              </div>
            )}
            {centerSubtext && (
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {centerSubtext}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
