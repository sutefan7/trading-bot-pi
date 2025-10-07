"""
Shadow ML Observer voor Trading Bot v4
Logt ML predictions zonder te handelen voor vergelijking
"""
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger


@dataclass
class MLPrediction:
    """ML prediction data"""
    timestamp: datetime
    symbol: str
    strategy: str
    ml_confidence: float
    ml_signal: str  # 'buy', 'sell', 'hold'
    traditional_signal: str  # 'buy', 'sell', 'hold'
    traditional_confidence: float
    price: float
    features: Dict[str, float]
    model_version: str
    latency_ms: float


@dataclass
class ShadowMLStats:
    """Shadow ML statistics"""
    total_predictions: int
    ml_accuracy: float
    traditional_accuracy: float
    ml_win_rate: float
    traditional_win_rate: float
    avg_ml_latency: float
    avg_traditional_latency: float
    correlation: float
    start_time: datetime
    end_time: Optional[datetime] = None


class ShadowMLObserver:
    """Observer voor shadow ML logging en vergelijking"""
    
    def __init__(self, output_dir: str = "storage/reports/shadow_ml"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.predictions: List[MLPrediction] = []
        self.stats: Optional[ShadowMLStats] = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.prediction_count = 0
        
        # Log file
        self.log_file = self.output_dir / f"shadow_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        logger.info(f"Shadow ML Observer initialized: {self.output_dir}")
    
    def log_prediction(self, symbol: str, strategy: str, 
                      ml_confidence: float, ml_signal: str,
                      traditional_signal: str, traditional_confidence: float,
                      price: float, features: Dict[str, float],
                      model_version: str, latency_ms: float):
        """
        Log een ML prediction voor vergelijking
        
        Args:
            symbol: Trading symbol
            strategy: Strategy naam
            ml_confidence: ML model confidence
            ml_signal: ML model signal
            traditional_signal: Traditional strategy signal
            traditional_confidence: Traditional strategy confidence
            price: Current price
            features: Feature values
            model_version: Model version
            latency_ms: Prediction latency in milliseconds
        """
        try:
            prediction = MLPrediction(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy=strategy,
                ml_confidence=ml_confidence,
                ml_signal=ml_signal,
                traditional_signal=traditional_signal,
                traditional_confidence=traditional_confidence,
                price=price,
                features=features,
                model_version=model_version,
                latency_ms=latency_ms
            )
            
            # Store prediction
            self.predictions.append(prediction)
            self.prediction_count += 1
            
            # Log to file
            self._log_to_file(prediction)
            
            # Log summary
            if self.prediction_count % 10 == 0:
                logger.info(f"Shadow ML: {self.prediction_count} predictions logged")
            
        except Exception as e:
            logger.error(f"Error logging ML prediction: {e}")
    
    def _log_to_file(self, prediction: MLPrediction):
        """Log prediction to JSONL file"""
        try:
            with open(self.log_file, 'a') as f:
                # Convert datetime to ISO string
                pred_dict = asdict(prediction)
                pred_dict['timestamp'] = prediction.timestamp.isoformat()
                
                f.write(json.dumps(pred_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
    
    def update_stats(self):
        """Update shadow ML statistics"""
        try:
            if not self.predictions:
                return
            
            # Calculate statistics
            total_predictions = len(self.predictions)
            
            # ML vs Traditional accuracy
            ml_correct = sum(1 for p in self.predictions if p.ml_signal == p.traditional_signal)
            ml_accuracy = ml_correct / total_predictions if total_predictions > 0 else 0
            
            # Win rates (simplified - would need actual trade outcomes)
            ml_win_rate = sum(1 for p in self.predictions if p.ml_confidence > 0.6) / total_predictions
            traditional_win_rate = sum(1 for p in self.predictions if p.traditional_confidence > 0.6) / total_predictions
            
            # Latency
            avg_ml_latency = sum(p.latency_ms for p in self.predictions) / total_predictions
            avg_traditional_latency = 0  # Traditional strategies don't have latency
            
            # Correlation
            ml_confidences = [p.ml_confidence for p in self.predictions]
            traditional_confidences = [p.traditional_confidence for p in self.predictions]
            correlation = self._calculate_correlation(ml_confidences, traditional_confidences)
            
            self.stats = ShadowMLStats(
                total_predictions=total_predictions,
                ml_accuracy=ml_accuracy,
                traditional_accuracy=ml_accuracy,  # Same as ML for now
                ml_win_rate=ml_win_rate,
                traditional_win_rate=traditional_win_rate,
                avg_ml_latency=avg_ml_latency,
                avg_traditional_latency=avg_traditional_latency,
                correlation=correlation,
                start_time=self.start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            sum_y2 = sum(y[i] ** 2 for i in range(n))
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def get_stats(self) -> Optional[ShadowMLStats]:
        """Get current statistics"""
        self.update_stats()
        return self.stats
    
    def export_report(self) -> str:
        """Export shadow ML report"""
        try:
            self.update_stats()
            
            if not self.stats:
                return "No statistics available"
            
            report_file = self.output_dir / f"shadow_ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Create report
            report = {
                "summary": asdict(self.stats),
                "predictions": [asdict(p) for p in self.predictions[-100:]],  # Last 100 predictions
                "export_time": datetime.now().isoformat()
            }
            
            # Convert datetime objects
            report["summary"]["start_time"] = self.stats.start_time.isoformat()
            if self.stats.end_time:
                report["summary"]["end_time"] = self.stats.end_time.isoformat()
            
            for pred in report["predictions"]:
                pred["timestamp"] = pred["timestamp"].isoformat()
            
            # Write report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Shadow ML report exported: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return ""
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            self.update_stats()
            
            if not self.stats:
                return {"error": "No statistics available"}
            
            return {
                "total_predictions": self.stats.total_predictions,
                "ml_accuracy": f"{self.stats.ml_accuracy:.2%}",
                "traditional_accuracy": f"{self.stats.traditional_accuracy:.2%}",
                "ml_win_rate": f"{self.stats.ml_win_rate:.2%}",
                "traditional_win_rate": f"{self.stats.traditional_win_rate:.2%}",
                "avg_ml_latency": f"{self.stats.avg_ml_latency:.2f}ms",
                "correlation": f"{self.stats.correlation:.3f}",
                "duration": str(self.stats.end_time - self.stats.start_time) if self.stats.end_time else "ongoing"
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def reset(self):
        """Reset observer data"""
        self.predictions.clear()
        self.stats = None
        self.start_time = datetime.now()
        self.prediction_count = 0
        
        # Create new log file
        self.log_file = self.output_dir / f"shadow_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        logger.info("Shadow ML Observer reset")
    
    def cleanup_old_logs(self, days: int = 7):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for log_file in self.output_dir.glob("shadow_ml_*.jsonl"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Removed old log file: {log_file}")
            
            for report_file in self.output_dir.glob("shadow_ml_report_*.json"):
                if report_file.stat().st_mtime < cutoff_date.timestamp():
                    report_file.unlink()
                    logger.info(f"Removed old report file: {report_file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")


# Example usage
if __name__ == "__main__":
    # Test shadow ML observer
    observer = ShadowMLObserver()
    
    # Simulate some predictions
    for i in range(5):
        observer.log_prediction(
            symbol="BTC-USD",
            strategy="trend_follow",
            ml_confidence=0.75,
            ml_signal="buy",
            traditional_signal="buy",
            traditional_confidence=0.65,
            price=50000.0,
            features={"rsi": 45, "macd": 0.1},
            model_version="v1.0",
            latency_ms=25.5
        )
    
    # Get stats
    stats = observer.get_stats()
    if stats:
        print(f"Stats: {stats}")
    
    # Export report
    report_file = observer.export_report()
    print(f"Report exported: {report_file}")
    
    # Get summary
    summary = observer.get_performance_summary()
    print(f"Summary: {summary}")
