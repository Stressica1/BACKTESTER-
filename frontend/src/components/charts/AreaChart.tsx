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
import { getAreaChartOptions, chartColors } from '@/lib/chartConfig';
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

interface AreaChartProps {
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
  gradientFill?: boolean;
}

export const AreaChart: React.FC<AreaChartProps> = ({
  data,
  height = '300px',
  showLegend = true,
  gradientFill = true,
}) => {
  const { theme } = useThemeStore();
  const chartOptions = getAreaChartOptions(theme);
  const canvasRef = React.useRef<any>(null);

  // Create gradient backgrounds
  const createGradient = (ctx: CanvasRenderingContext2D, color: string) => {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, `${color}60`);
    gradient.addColorStop(1, `${color}10`);
    return gradient;
  };

  const processedData = React.useMemo(() => {
    if (!canvasRef.current || !gradientFill) {
      return {
        ...data,
        datasets: data.datasets.map((dataset) => ({
          ...dataset,
          fill: true,
          borderColor: dataset.borderColor || chartColors.primary,
          backgroundColor: dataset.backgroundColor || `${chartColors.primary}20`,
          tension: dataset.tension || 0.4,
        })),
      };
    }

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return data;

    return {
      ...data,
      datasets: data.datasets.map((dataset) => ({
        ...dataset,
        fill: true,
        borderColor: dataset.borderColor || chartColors.primary,
        backgroundColor: gradientFill 
          ? createGradient(ctx, dataset.borderColor || chartColors.primary)
          : dataset.backgroundColor || `${chartColors.primary}20`,
        tension: dataset.tension || 0.4,
        pointBackgroundColor: dataset.borderColor || chartColors.primary,
        pointBorderColor: dataset.borderColor || chartColors.primary,
        pointHoverBackgroundColor: dataset.borderColor || chartColors.primary,
        pointHoverBorderColor: '#FFFFFF',
      })),
    };
  }, [data, gradientFill]);

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
    <div style={{ height }} className="will-change-transform">
      <Line
        ref={canvasRef}
        data={processedData}
        options={options}
      />
    </div>
  );
};
