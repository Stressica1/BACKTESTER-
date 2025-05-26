// Chart components for the trading platform
export { LineChart } from './LineChart';
export { BarChart } from './BarChart';
export { AreaChart } from './AreaChart';
export { DoughnutChart } from './DoughnutChart';
export { CandlestickChart } from './CandlestickChart';
export { RiskGauge } from './RiskGauge';
export { Heatmap } from './Heatmap';

// Chart configuration and utilities
export {
  chartColors,
  chartThemes,
  getBaseChartOptions,
  getLineChartOptions,
  getBarChartOptions,
  getDoughnutChartOptions,
  getAreaChartOptions,
  getCandlestickChartOptions,
  getRiskGaugeOptions,
  getHeatmapColors,
} from '@/lib/chartConfig';
