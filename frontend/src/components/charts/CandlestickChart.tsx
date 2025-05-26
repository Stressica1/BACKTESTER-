'use client';

import React from 'react';
import {
  Chart as ChartJS,
  LinearScale,
  CategoryScale,
  BarElement,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
  LineController,
  BarController,
} from 'chart.js';
import dynamic from 'next/dynamic';
import { chartColors } from '@/lib/chartConfig';
import { useThemeStore } from '@/lib/store';

ChartJS.register(
  LinearScale,
  CategoryScale,
  BarElement,
  PointElement,
  LineElement,
  Legend,
  Tooltip,
  LineController,
  BarController
);

// Dynamically load chart component to reduce bundle size and disable SSR
const Chart = dynamic(
  () => import('react-chartjs-2').then((mod) => mod.Chart),
  {
    ssr: false,
    loading: () => <div className="animate-pulse bg-gray-200 h-full w-full rounded-lg" />
  }
);

interface CandlestickData {
  x: string | Date;
  o: number; // open
  h: number; // high
  l: number; // low
  c: number; // close
  v?: number; // volume
}

interface CandlestickChartProps {
  data: CandlestickData[];
  height?: string;
  showVolume?: boolean;
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  height = '400px',
  showVolume = true,
}) => {
  const { theme } = useThemeStore();

  // Transform candlestick data for Chart.js
  const chartData = {
    datasets: [
      {
        label: 'Price',
        data: data.map(item => ({
          x: item.x,
          o: item.o,
          h: item.h,
          l: item.l,
          c: item.c,
        })),
        type: 'candlestick' as const,
        borderColor: chartColors.primary,
        backgroundColor: chartColors.success,
        downBackgroundColor: chartColors.danger,
        downBorderColor: chartColors.danger,
      },
      ...(showVolume ? [{
        label: 'Volume',
        data: data.map((item, index) => ({
          x: item.x,
          y: item.v || 0,
        })),
        type: 'bar' as const,
        backgroundColor: data.map(item => 
          item.c >= item.o ? `${chartColors.success}40` : `${chartColors.danger}40`
        ),
        borderColor: data.map(item => 
          item.c >= item.o ? chartColors.success : chartColors.danger
        ),
        borderWidth: 1,
        yAxisID: 'volume',
        order: 2,
      }] : []),
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: theme === 'dark' ? 'rgba(17, 24, 39, 0.95)' : 'rgba(255, 255, 255, 0.95)',
        titleColor: theme === 'dark' ? '#F9FAFB' : '#374151',
        bodyColor: theme === 'dark' ? '#F9FAFB' : '#374151',
        borderColor: theme === 'dark' ? '#4B5563' : '#D1D5DB',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        callbacks: {
          title: (context: any) => {
            return new Date(context[0].label).toLocaleString();
          },
          beforeBody: (context: any) => {
            const candleData = context.find((ctx: any) => ctx.dataset.type === 'candlestick');
            if (candleData && candleData.raw) {
              const { o, h, l, c } = candleData.raw;
              return [
                `Open: $${o.toFixed(2)}`,
                `High: $${h.toFixed(2)}`,
                `Low: $${l.toFixed(2)}`,
                `Close: $${c.toFixed(2)}`,
                `Change: ${c >= o ? '+' : ''}${((c - o) / o * 100).toFixed(2)}%`,
              ];
            }
            return [];
          },
          label: (context: any) => {
            if (context.dataset.type === 'bar' && showVolume) {
              return `Volume: ${context.parsed.y.toLocaleString()}`;
            }
            return '';
          },
        },
      },
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM dd',
            week: 'MMM dd',
            month: 'MMM yyyy',
          },
        },
        grid: {
          color: theme === 'dark' ? '#374151' : '#E5E7EB',
          drawBorder: false,
        },
        ticks: {
          color: theme === 'dark' ? '#F9FAFB' : '#374151',
          font: {
            size: 11,
          },
        },
      },
      y: {
        type: 'linear' as const,
        position: 'right' as const,
        grid: {
          color: theme === 'dark' ? '#374151' : '#E5E7EB',
          drawBorder: false,
        },
        ticks: {
          color: theme === 'dark' ? '#F9FAFB' : '#374151',
          font: {
            size: 11,
          },
          callback: function(value: any) {
            return `$${value.toFixed(2)}`;
          },
        },
      },
      ...(showVolume ? {
        volume: {
          type: 'linear' as const,
          position: 'left' as const,
          max: Math.max(...data.map(d => d.v || 0)) * 4, // Scale volume to 25% of chart height
          grid: {
            display: false,
          },
          ticks: {
            display: false,
          },
        },
      } : {}),
    },
  };

  return (
    <div style={{ height }} className="will-change-transform">
      <Chart type="line" data={chartData} options={options} />
    </div>
  );
};
