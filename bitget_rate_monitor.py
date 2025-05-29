#!/usr/bin/env python3
"""
Bitget API Rate Limit Monitor and Reporter

This module provides comprehensive monitoring of Bitget API rate limits,
generating detailed reports and alerts for rate limit compliance.
"""

import json
import time
import datetime
import logging
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BitgetRateMonitor")

@dataclass
class RateLimitEvent:
    """Rate limit event data structure"""
    timestamp: float
    endpoint: str
    endpoint_type: str  # 'public' or 'private'
    success: bool
    error_code: Optional[str] = None
    retry_count: int = 0
    backoff_time: float = 0.0

@dataclass
class EndpointStats:
    """Statistics for an API endpoint"""
    endpoint: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limit_hits: int = 0
    avg_response_time: float = 0.0
    last_request_time: float = 0.0
    requests_per_second: float = 0.0
    compliance_score: float = 100.0

class BitgetRateMonitor:
    """Comprehensive Bitget API rate limit monitoring system"""
    
    def __init__(self, log_file='bitget_rate_monitor.log'):
        self.log_file = log_file
        self.events = deque(maxlen=10000)  # Store last 10k events
        self.endpoint_stats = defaultdict(lambda: EndpointStats(""))
        self.rate_limit_windows = {
            'public': deque(maxlen=20),    # 20 req/s for public
            'private': deque(maxlen=10),   # 10 req/s for private
            'batch': deque(maxlen=5)       # 5 req/s for batch
        }
        
        # Bitget rate limits from official documentation
        self.rate_limits = {
            'public': {
                'time': 20, 'currencies': 20, 'products': 20, 'product': 20,
                'ticker': 20, 'tickers': 20, 'fills': 20, 'candles': 20,
                'depth': 20, 'transferRecords': 20, 'orderInfo': 20,
                'open-orders': 20, 'history': 20, 'trade_fills': 20
            },
            'private': {
                'assets': 10, 'bills': 10, 'orders': 10, 'cancel-order': 10,
                'cancel-batch-orders': 10, 'batch-orders': 5
            }
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'rate_limit_hits_per_minute': 5,
            'consecutive_failures': 3,
            'compliance_score_threshold': 85.0,
            'requests_per_second_warning': 0.8  # 80% of limit
        }
        
        # Setup file logging
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(self.file_handler)
        
    def record_request(self, endpoint: str, endpoint_type: str, success: bool = True, 
                      error_code: Optional[str] = None, response_time: float = 0.0,
                      retry_count: int = 0, backoff_time: float = 0.0):
        """Record an API request event"""
        timestamp = time.time()
        
        # Create event
        event = RateLimitEvent(
            timestamp=timestamp,
            endpoint=endpoint,
            endpoint_type=endpoint_type,
            success=success,
            error_code=error_code,
            retry_count=retry_count,
            backoff_time=backoff_time
        )
        
        # Store event
        self.events.append(event)
        
        # Update rate limit windows
        if endpoint_type in self.rate_limit_windows:
            self.rate_limit_windows[endpoint_type].append(timestamp)
        
        # Update endpoint statistics
        stats = self.endpoint_stats[endpoint]
        if not stats.endpoint:
            stats.endpoint = endpoint
        
        stats.total_requests += 1
        stats.last_request_time = timestamp
        
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
            
        if error_code and ('429' in str(error_code) or '30007' in str(error_code)):
            stats.rate_limit_hits += 1
            
        # Update average response time
        if response_time > 0:
            stats.avg_response_time = (
                (stats.avg_response_time * (stats.total_requests - 1) + response_time) /
                stats.total_requests
            )
        
        # Calculate requests per second (last 60 seconds)
        recent_requests = [
            e for e in self.events 
            if e.endpoint == endpoint and timestamp - e.timestamp <= 60
        ]
        stats.requests_per_second = len(recent_requests) / 60.0
        
        # Calculate compliance score
        stats.compliance_score = self._calculate_compliance_score(endpoint, stats)
        
        # Log the event
        self._log_event(event, stats)
        
        # Check for alerts
        self._check_alerts(endpoint, stats, event)
    
    def _calculate_compliance_score(self, endpoint: str, stats: EndpointStats) -> float:
        """Calculate compliance score for an endpoint (0-100)"""
        score = 100.0
        
        # Deduct points for rate limit hits
        if stats.total_requests > 0:
            rate_limit_ratio = stats.rate_limit_hits / stats.total_requests
            score -= (rate_limit_ratio * 50)  # Up to 50 points for rate limits
        
        # Deduct points for failed requests
        if stats.total_requests > 0:
            failure_ratio = stats.failed_requests / stats.total_requests
            score -= (failure_ratio * 30)  # Up to 30 points for failures
        
        # Deduct points for high request rate
        endpoint_type = 'public' if endpoint in self.rate_limits['public'] else 'private'
        limit = self.rate_limits[endpoint_type].get(endpoint, 10)
        if stats.requests_per_second > limit * 0.8:  # 80% of limit
            over_rate = (stats.requests_per_second / limit) - 0.8
            score -= (over_rate * 20)  # Up to 20 points for high rate
        
        return max(0.0, min(100.0, score))
    
    def _log_event(self, event: RateLimitEvent, stats: EndpointStats):
        """Log an event to file and console"""
        if not event.success:
            if event.error_code and ('429' in str(event.error_code) or '30007' in str(event.error_code)):
                logger.error(
                    f"RATE LIMIT HIT: {event.endpoint} - Error: {event.error_code} - "
                    f"Retry: {event.retry_count} - Backoff: {event.backoff_time:.2f}s - "
                    f"RPS: {stats.requests_per_second:.2f} - Score: {stats.compliance_score:.1f}"
                )
            else:
                logger.warning(
                    f"REQUEST FAILED: {event.endpoint} - Error: {event.error_code} - "
                    f"Score: {stats.compliance_score:.1f}"
                )
        else:
            logger.debug(
                f"REQUEST OK: {event.endpoint} - RPS: {stats.requests_per_second:.2f} - "
                f"Score: {stats.compliance_score:.1f}"
            )
    
    def _check_alerts(self, endpoint: str, stats: EndpointStats, event: RateLimitEvent):
        """Check for alert conditions and send alerts"""
        current_time = time.time()
        
        # Alert for rate limit hits in last minute
        recent_rate_limits = [
            e for e in self.events 
            if e.endpoint == endpoint and 
               current_time - e.timestamp <= 60 and 
               not e.success and e.error_code and 
               ('429' in str(e.error_code) or '30007' in str(e.error_code))
        ]
        
        if len(recent_rate_limits) >= self.alert_thresholds['rate_limit_hits_per_minute']:
            self._send_alert(
                'CRITICAL', 
                f"Excessive rate limiting on {endpoint}: {len(recent_rate_limits)} hits in 1 minute"
            )
        
        # Alert for consecutive failures
        recent_events = [
            e for e in reversed(list(self.events)) 
            if e.endpoint == endpoint
        ][:self.alert_thresholds['consecutive_failures']]
        
        if (len(recent_events) >= self.alert_thresholds['consecutive_failures'] and 
            all(not e.success for e in recent_events)):
            self._send_alert(
                'ERROR',
                f"Consecutive failures on {endpoint}: {len(recent_events)} failed requests"
            )
        
        # Alert for low compliance score
        if stats.compliance_score < self.alert_thresholds['compliance_score_threshold']:
            self._send_alert(
                'WARNING',
                f"Low compliance score for {endpoint}: {stats.compliance_score:.1f}%"
            )
        
        # Alert for high request rate
        endpoint_type = 'public' if endpoint in self.rate_limits['public'] else 'private'
        limit = self.rate_limits[endpoint_type].get(endpoint, 10)
        warning_threshold = limit * self.alert_thresholds['requests_per_second_warning']
        
        if stats.requests_per_second > warning_threshold:
            self._send_alert(
                'WARNING',
                f"High request rate for {endpoint}: {stats.requests_per_second:.2f} req/s "
                f"(limit: {limit} req/s)"
            )
    
    def _send_alert(self, level: str, message: str):
        """Send an alert (log for now, can be extended to email/slack/etc)"""
        logger.log(getattr(logging, level), f"🚨 ALERT: {message}")
        
        # Write to alert file
        with open('bitget_alerts.log', 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {level} - {message}\n")
    
    def get_current_status(self) -> Dict:
        """Get current rate limiting status"""
        current_time = time.time()
        
        # Calculate current requests per second for each window
        status = {
            'timestamp': current_time,
            'windows': {},
            'top_endpoints': [],
            'alerts': {
                'active_rate_limits': 0,
                'recent_failures': 0,
                'low_compliance_endpoints': []
            }
        }
        
        # Check rate limit windows
        for window_type, timestamps in self.rate_limit_windows.items():
            recent = [t for t in timestamps if current_time - t <= 1.0]
            status['windows'][window_type] = {
                'requests_last_second': len(recent),
                'limit': 20 if window_type == 'public' else (10 if window_type == 'private' else 5),
                'utilization': len(recent) / (20 if window_type == 'public' else (10 if window_type == 'private' else 5))
            }
        
        # Get top endpoints by volume
        sorted_endpoints = sorted(
            self.endpoint_stats.values(),
            key=lambda x: x.total_requests,
            reverse=True
        )
        
        status['top_endpoints'] = [
            {
                'endpoint': stats.endpoint,
                'total_requests': stats.total_requests,
                'success_rate': stats.successful_requests / max(stats.total_requests, 1) * 100,
                'rate_limit_hits': stats.rate_limit_hits,
                'requests_per_second': stats.requests_per_second,
                'compliance_score': stats.compliance_score
            }
            for stats in sorted_endpoints[:10]
        ]
        
        # Check for active alerts
        for stats in self.endpoint_stats.values():
            if stats.rate_limit_hits > 0:
                recent_rate_limits = [
                    e for e in self.events 
                    if e.endpoint == stats.endpoint and 
                       current_time - e.timestamp <= 60 and 
                       not e.success and e.error_code and 
                       ('429' in str(e.error_code) or '30007' in str(e.error_code))
                ]
                status['alerts']['active_rate_limits'] += len(recent_rate_limits)
            
            if stats.compliance_score < self.alert_thresholds['compliance_score_threshold']:
                status['alerts']['low_compliance_endpoints'].append({
                    'endpoint': stats.endpoint,
                    'score': stats.compliance_score
                })
        
        return status
    
    def generate_report(self, hours: int = 24) -> str:
        """Generate a comprehensive rate limiting report"""
        current_time = time.time()
        start_time = current_time - (hours * 3600)
        
        # Filter events for time period
        period_events = [e for e in self.events if e.timestamp >= start_time]
        
        report = f"""
# Bitget API Rate Limit Report
## Period: Last {hours} hours ({datetime.datetime.fromtimestamp(start_time)} to {datetime.datetime.fromtimestamp(current_time)})

### Summary
- Total Requests: {len(period_events)}
- Successful Requests: {sum(1 for e in period_events if e.success)}
- Failed Requests: {sum(1 for e in period_events if not e.success)}
- Rate Limit Hits: {sum(1 for e in period_events if not e.success and e.error_code and ('429' in str(e.error_code) or '30007' in str(e.error_code)))}
- Average Requests/Hour: {len(period_events) / hours:.1f}

### Endpoint Statistics
"""
        
        # Create endpoint statistics table
        endpoint_data = []
        for stats in sorted(self.endpoint_stats.values(), key=lambda x: x.total_requests, reverse=True):
            endpoint_events = [e for e in period_events if e.endpoint == stats.endpoint]
            if endpoint_events:
                endpoint_data.append({
                    'Endpoint': stats.endpoint,
                    'Requests': len(endpoint_events),
                    'Success Rate': f"{sum(1 for e in endpoint_events if e.success) / len(endpoint_events) * 100:.1f}%",
                    'Rate Limits': sum(1 for e in endpoint_events if not e.success and e.error_code and ('429' in str(e.error_code) or '30007' in str(e.error_code))),
                    'Req/Sec': f"{stats.requests_per_second:.2f}",
                    'Compliance': f"{stats.compliance_score:.1f}%"
                })
        
        if endpoint_data:
            df = pd.DataFrame(endpoint_data)
            report += df.to_string(index=False)
        
        report += f"""

### Rate Limit Compliance
- Endpoints with compliance < 90%: {sum(1 for stats in self.endpoint_stats.values() if stats.compliance_score < 90)}
- Endpoints with rate limit hits: {sum(1 for stats in self.endpoint_stats.values() if stats.rate_limit_hits > 0)}

### Recommendations
"""
        
        # Add recommendations
        recommendations = []
        for stats in self.endpoint_stats.values():
            if stats.compliance_score < 85:
                recommendations.append(f"- {stats.endpoint}: Implement stricter rate limiting (score: {stats.compliance_score:.1f}%)")
            
            endpoint_type = 'public' if stats.endpoint in self.rate_limits['public'] else 'private'
            limit = self.rate_limits[endpoint_type].get(stats.endpoint, 10)
            if stats.requests_per_second > limit * 0.8:
                recommendations.append(f"- {stats.endpoint}: Reduce request frequency (current: {stats.requests_per_second:.2f} req/s, limit: {limit} req/s)")
        
        if recommendations:
            report += "\n".join(recommendations)
        else:
            report += "- All endpoints are operating within acceptable limits"
        
        return report
    
    def export_data(self, filename: str = None) -> str:
        """Export monitoring data to JSON file"""
        if filename is None:
            filename = f"bitget_rate_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'export_time': time.time(),
            'events': [asdict(event) for event in self.events],
            'endpoint_stats': {k: asdict(v) for k, v in self.endpoint_stats.items()},
            'rate_limits': self.rate_limits,
            'alert_thresholds': self.alert_thresholds
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Rate monitoring data exported to {filename}")
        return filename

# Global monitor instance
rate_monitor = BitgetRateMonitor()

def record_api_request(endpoint: str, endpoint_type: str = 'public', success: bool = True, 
                      error_code: Optional[str] = None, response_time: float = 0.0,
                      retry_count: int = 0, backoff_time: float = 0.0):
    """Convenience function to record API requests"""
    rate_monitor.record_request(endpoint, endpoint_type, success, error_code, 
                               response_time, retry_count, backoff_time)

def get_rate_status() -> Dict:
    """Get current rate limiting status"""
    return rate_monitor.get_current_status()

def generate_rate_report(hours: int = 24) -> str:
    """Generate rate limiting report"""
    return rate_monitor.generate_report(hours)

if __name__ == "__main__":
    # Example usage
    monitor = BitgetRateMonitor()
    
    # Simulate some API calls
    import random
    for i in range(100):
        endpoint = random.choice(['ticker', 'orders', 'candles', 'depth'])
        endpoint_type = 'public' if endpoint in ['ticker', 'candles', 'depth'] else 'private'
        success = random.random() > 0.1  # 90% success rate
        error_code = '429' if not success and random.random() > 0.5 else None
        
        monitor.record_request(endpoint, endpoint_type, success, error_code)
        time.sleep(0.1)
    
    # Generate report
    report = monitor.generate_report(hours=1)
    print(report)
    
    # Export data
    filename = monitor.export_data()
    print(f"\nData exported to: {filename}") 