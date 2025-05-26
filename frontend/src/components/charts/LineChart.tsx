'use client';

import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import dynamic from 'next/dynamic';
import { getLineChartOptions, chartColors } from '@/lib/chartConfig';
import { useThemeStore } from '@/lib/store';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Dynamically load chart component to reduce bundle size and disable SSR
const Line = dynamic(
  () => import('react-chartjs-2').then((mod) => mod.Line),
  {
    ssr: false,
    loading: () => <div className="animate-pulse bg-gray-200 h-full w-full rounded-lg" />
  }
);

interface LineChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      borderColor?: string;
      backgroundColor?: string;
      fill?: boolean;
      tension?: number;
    }[];
  };
  height?: string;
  showLegend?: boolean;
  showTooltip?: boolean;
}

export const LineChart: React.FC<LineChartProps> = ({
  data,
  height = '300px',
  showLegend = true,
  showTooltip = true,
}) => {
  const { theme } = useThemeStore();
  const chartOptions = getLineChartOptions(theme);

  // Apply default styling to datasets
  const processedData = {
    ...data,
    datasets: data.datasets.map((dataset, index) => ({
      ...dataset,
      borderColor: dataset.borderColor || chartColors.primary,
      backgroundColor: dataset.backgroundColor || 
        (dataset.fill ? `${chartColors.primary}20` : chartColors.primary),
      pointBackgroundColor: dataset.borderColor || chartColors.primary,
      pointBorderColor: dataset.borderColor || chartColors.primary,
      pointHoverBackgroundColor: dataset.borderColor || chartColors.primary,
      pointHoverBorderColor: '#FFFFFF',
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
      tooltip: {
        ...chartOptions.plugins.tooltip,
        enabled: showTooltip,
      },
    },
  };

  return (
    <div style={{ height }} className="will-change-transform">
      <Line data={processedData} options={options} />
    </div>
  );
};
