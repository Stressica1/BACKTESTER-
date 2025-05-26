'use client';

import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import dynamic from 'next/dynamic';
import { getBarChartOptions, chartColors } from '@/lib/chartConfig';
import { useThemeStore } from '@/lib/store';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Dynamically load chart component to reduce bundle size and disable SSR
const Bar = dynamic(
  () => import('react-chartjs-2').then((mod) => mod.Bar),
  {
    ssr: false,
    loading: () => <div className="animate-pulse bg-gray-200 h-full w-full rounded-lg" />
  }
);

interface BarChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor?: string | string[];
      borderColor?: string | string[];
      borderWidth?: number;
    }[];
  };
  height?: string;
  horizontal?: boolean;
  showLegend?: boolean;
}

export const BarChart: React.FC<BarChartProps> = ({
  data,
  height = '300px',
  horizontal = false,
  showLegend = true,
}) => {
  const { theme } = useThemeStore();
  const chartOptions = getBarChartOptions(theme);

  // Apply default styling to datasets
  const processedData = {
    ...data,
    datasets: data.datasets.map((dataset, index) => ({
      ...dataset,
      backgroundColor: dataset.backgroundColor || chartColors.primary,
      borderColor: dataset.borderColor || chartColors.primary,
      borderWidth: dataset.borderWidth || 0,
    })),
  };

  const options = {
    ...chartOptions,
    indexAxis: horizontal ? ('y' as const) : ('x' as const),
    plugins: {
      ...chartOptions.plugins,
      legend: {
        ...chartOptions.plugins.legend,
        display: showLegend,
      },
    },
  };

  return (
    <div style={{ height }} className="will-change-transform">
      <Bar data={processedData} options={options} />
    </div>
  );
};
