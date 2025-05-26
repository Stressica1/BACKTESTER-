'use client';

import React from 'react';

interface HeatmapData {
  value: number;
  label: string;
  change?: number;
}

interface HeatmapProps {
  data: HeatmapData[];
  height?: string;
  cellSize?: number;
  columns?: number;
}

export const Heatmap: React.FC<HeatmapProps> = ({
  data,
  height = '300px',
  cellSize = 80,
  columns = 5,
}) => {
  // Calculate color based on value change
  const getColor = (change: number = 0) => {
    const intensity = Math.min(Math.abs(change) / 10, 1); // Normalize to 0-1
    
    if (change > 0) {
      // Green for positive
      const opacity = 0.2 + (intensity * 0.6);
      return `rgba(16, 185, 129, ${opacity})`;
    } else if (change < 0) {
      // Red for negative
      const opacity = 0.2 + (intensity * 0.6);
      return `rgba(239, 68, 68, ${opacity})`;
    } else {
      // Gray for neutral
      return 'rgba(107, 114, 128, 0.2)';
    }
  };

  const getTextColor = (change: number = 0) => {
    const intensity = Math.abs(change) / 10;
    if (intensity > 0.5) {
      return 'white';
    }
    return undefined; // Use default text color
  };

  return (
    <div 
      style={{ height }}
      className="p-4 overflow-auto"
    >
      <div 
        className="grid gap-2"
        style={{ 
          gridTemplateColumns: `repeat(${columns}, 1fr)`,
          minHeight: 'fit-content',
        }}
      >
        {data.map((item, index) => (
          <div
            key={index}
            className="relative flex flex-col items-center justify-center p-3 rounded-lg border border-gray-200 dark:border-gray-700 transition-all duration-200 hover:scale-105 cursor-pointer"
            style={{
              backgroundColor: getColor(item.change),
              color: getTextColor(item.change),
              minWidth: cellSize,
              minHeight: cellSize,
            }}
          >
            <div className="text-xs font-medium text-center mb-1 truncate w-full">
              {item.label}
            </div>
            <div className="text-lg font-bold">
              ${item.value.toLocaleString()}
            </div>
            {item.change !== undefined && (
              <div className={`text-xs font-medium ${
                item.change >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}%
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
