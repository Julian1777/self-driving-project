import numpy as np
import requests
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][Aggregator] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class AggregationResult:
    """Result object returned from aggregator.process_frame()"""
    def __init__(self, results, processing_time_ms, service_status, all_healthy):
        self.results = results  # Results keyed by service name, None if failed
        self.processing_time_ms = processing_time_ms  # Total time to get all results
        self.service_status = service_status  # Status of each service ('ok', 'timeout', 'error', etc)
        self.all_healthy = all_healthy  # True if all services succeeded


class PerceptionAggregator:
    """
    Orchestrates concurrent calls to all perception microservices.
    
    Typical usage:
    ```
    aggregator = PerceptionAggregator({
        'lane_detection': 'http://localhost:4777',
        'object_detection': 'http://localhost:5777',
        ...
    })
    
    result = aggregator.process_frame(frame, speed_kph, timestamp_ns)
    ```
    """
    
    def __init__(self, service_config, timeout=2.0, max_workers=None, retry_count=1):
        """
        Initialize the aggregator.
        
        Args:
            service_config: Dict mapping service names to their HTTP endpoints
                {
                    'lane_detection': 'http://localhost:4777',
                    'object_detection': 'http://localhost:5777',
                    'traffic_light_detection': 'http://localhost:6777',
                    'sign_detection': 'http://localhost:7777',
                    'sign_classification': 'http://localhost:8777'
                }
            timeout: Max seconds to wait for each service response
            max_workers: Number of concurrent threads (default: number of services)
            retry_count: Number of retries for failed requests
        """
        self.service_config = service_config
        self.timeout = timeout
        self.retry_count = retry_count
        
        # Set max_workers to number of services if not specified
        if max_workers is None:
            max_workers = len(service_config)
        
        # ThreadPoolExecutor manages concurrent requests
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='perception-'
        )
        
        logger.info(f"Aggregator initialized with {len(service_config)} services")
        logger.info(f"Service endpoints: {list(service_config.keys())}")
        logger.info(f"Timeout: {timeout}s, Max workers: {max_workers}")
    
    def health_check(self):
        """
        Check health of all services by calling their /health endpoints.
        
        Returns:
            Dict mapping service names to health status (True if healthy)
        """
        logger.info("Running health check on all services...")
        health_status = {}
        
        for service_name, endpoint in self.service_config.items():
            try:
                response = requests.get(
                    f"{endpoint}/health",
                    timeout=2.0
                )
                is_healthy = response.status_code == 200
                health_status[service_name] = is_healthy
                
                if is_healthy:
                    logger.info(f"✓ {service_name}: HEALTHY")
                else:
                    logger.warning(f"✗ {service_name}: UNHEALTHY (HTTP {response.status_code})")
                    
            except requests.ConnectionError:
                logger.error(f"✗ {service_name}: CONNECTION FAILED")
                health_status[service_name] = False
                
            except Exception as e:
                logger.error(f"✗ {service_name}: {type(e).__name__}: {e}")
                health_status[service_name] = False
        
        all_healthy = all(health_status.values())
        logger.info(f"Health check complete: {sum(health_status.values())}/{len(health_status)} healthy")
        
        return health_status
    
    def process_frame(self, frame, speed_kph, timestamp_ns, vehicle_pos=None, vehicle_direction=None):
        """
        Process a frame by sending it to all perception services concurrently.
        
        Args:
            frame: Camera frame as numpy array (H x W x 3, uint8)
            speed_kph: Vehicle speed in kilometers per hour
            timestamp_ns: Timestamp in nanoseconds
            vehicle_pos: Optional (x, y, z) position
            vehicle_direction: Optional (dx, dy, dz) direction vector
        
        Returns:
            AggregationResult with all service results
        
        Raises:
            ValueError: If frame shape is invalid
        """
        # Validate input
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be numpy array, got {type(frame)}")
        
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be HxWx3, got shape {frame.shape}")
        
        start_time = time.time()
        
        # Step 1: Prepare payload (convert numpy → JSON-compatible)
        logger.debug(f"Preparing payload for {frame.shape} frame...")
        payload = self._prepare_payload(frame, speed_kph, timestamp_ns, vehicle_pos, vehicle_direction)
        
        # Step 2: Submit concurrent tasks to all services
        logger.debug(f"Submitting {len(self.service_config)} concurrent requests...")
        futures = self._submit_tasks(payload)
        
        # Step 3: Collect results as they complete
        logger.debug("Waiting for service responses...")
        results = self._collect_results(futures)
        
        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000
        results['processing_time_ms'] = processing_time_ms
        
        logger.info(
            f"Aggregation complete in {processing_time_ms:.1f}ms | "
            f"Status: {results['service_status']} | "
            f"Healthy: {results['all_healthy']}"
        )
        
        return AggregationResult(**results)
    
    def _prepare_payload(self, frame, speed_kph, timestamp_ns, vehicle_pos=None, vehicle_direction=None):
        """
        Convert numpy frame to JSON-serializable payload.
        
        This is the critical bottleneck - we convert the numpy array to a list.
        
        Args:
            frame: numpy array
            speed_kph: vehicle speed
            timestamp_ns: timestamp
            vehicle_pos: optional position
            vehicle_direction: optional direction
        
        Returns:
            Dict ready to send via JSON over HTTP
        """
        # Convert numpy array to list
        # This is necessary because JSON doesn't natively support numpy types
        logger.debug(f"Converting frame {frame.shape} to list...")
        start_convert = time.time()
        
        frame_list = frame.tolist()  # Convert to nested lists
        
        convert_time_ms = (time.time() - start_convert) * 1000
        logger.debug(f"Frame conversion took {convert_time_ms:.1f}ms")
        
        # Build payload
        payload = {
            "frame": frame_list,
            "frame_shape": list(frame.shape),
            "speed_kph": float(speed_kph),
            "timestamp_ns": int(timestamp_ns)
        }
        
        # Optional fields
        if vehicle_pos is not None:
            payload["vehicle_pos"] = list(vehicle_pos)
        if vehicle_direction is not None:
            payload["vehicle_direction"] = list(vehicle_direction)
        
        return payload
    
    def _submit_tasks(self, payload):
        """
        Submit HTTP requests to all services concurrently.
        
        This uses ThreadPoolExecutor to make all requests at the same time.
        
        Args:
            payload: The frame data to send to all services
        
        Returns:
            Dict mapping service names to Future objects
        """
        futures = {}
        
        for service_name, endpoint in self.service_config.items():
            # Submit the request to thread pool
            future = self.executor.submit(
                self._make_request,
                service_name,
                f"{endpoint}/process",
                payload
            )
            futures[service_name] = future
            logger.debug(f"Submitted request to {service_name}")
        
        return futures
    
    def _make_request(self, service_name, url, payload):
        """
        Make HTTP request to a service with retry logic.
        
        Args:
            service_name: Name of the service (for logging)
            url: Full URL endpoint
            payload: JSON payload to send
        
        Returns:
            Response object or None if failed
        """
        for attempt in range(self.retry_count):
            try:
                logger.debug(f"[{service_name}] Attempt {attempt + 1}/{self.retry_count}")
                
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                logger.debug(f"[{service_name}] Response received: {response.status_code}")
                return response
                
            except requests.Timeout:
                logger.warning(f"[{service_name}] Timeout on attempt {attempt + 1}")
                if attempt < self.retry_count - 1:
                    logger.debug(f"[{service_name}] Retrying...")
                    time.sleep(0.1)  # Small delay before retry
                    
            except requests.ConnectionError as e:
                logger.warning(f"[{service_name}] Connection error: {e}")
                if attempt < self.retry_count - 1:
                    logger.debug(f"[{service_name}] Retrying...")
                    time.sleep(0.1)
                    
            except requests.HTTPError as e:
                logger.error(f"[{service_name}] HTTP error: {e}")
                return None
                
            except Exception as e:
                logger.error(f"[{service_name}] Unexpected error: {type(e).__name__}: {e}")
                return None
        
        # All retries failed
        logger.error(f"[{service_name}] Failed after {self.retry_count} attempts")
        return None
    
    def _collect_results(self, futures):
        """
        Collect results from all concurrent requests.
        
        This waits for all futures to complete (or timeout) and gathers their responses.
        
        Args:
            futures: Dict mapping service names to Future objects from ThreadPoolExecutor
        
        Returns:
            Dict with:
                - results: results keyed by service name (None if failed)
                - service_status: status of each service
                - all_healthy: boolean if all succeeded
        """
        results = {}
        service_status = {}
        
        # Iterate through futures and collect results
        for service_name, future in futures.items():
            try:
                # This blocks until the future completes (or we've already waited self.timeout)
                response = future.result(timeout=1.0)  # 1s timeout for getting result from future
                
                if response is None:
                    # Request failed during _make_request
                    logger.debug(f"[{service_name}] Request returned None")
                    results[service_name] = None
                    service_status[service_name] = 'failed'
                    
                else:
                    # Parse response JSON
                    try:
                        json_response = response.json()
                        results[service_name] = json_response
                        service_status[service_name] = 'ok'
                        logger.debug(f"[{service_name}] Successfully processed")
                        
                    except ValueError as e:
                        logger.error(f"[{service_name}] Invalid JSON response: {e}")
                        results[service_name] = None
                        service_status[service_name] = 'invalid_json'
                        
            except TimeoutError:
                logger.error(f"[{service_name}] Future result timeout")
                results[service_name] = None
                service_status[service_name] = 'timeout'
                
            except Exception as e:
                logger.error(f"[{service_name}] Error collecting result: {type(e).__name__}: {e}")
                results[service_name] = None
                service_status[service_name] = 'error'
        
        # Determine overall health
        all_healthy = all(status == 'ok' for status in service_status.values())
        
        return {
            'results': results,
            'service_status': service_status,
            'all_healthy': all_healthy
        }
    
    def shutdown(self):
        """Shutdown the thread pool executor gracefully."""
        logger.info("Shutting down aggregator...")
        self.executor.shutdown(wait=True)
        logger.info("Aggregator shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except:
            pass


# Convenience function for easy aggregator creation
def create_aggregator(host='localhost', port_base=4777, timeout=2.0):
    """
    Create an aggregator with default service configuration.
    
    Services are expected to be running at:
    - CV Lane Detection: host:4777
    - Object detection: host:5777
    - Traffic light: host:6777
    - Sign detection: host:7777
    - Sign classification: host:8777
    - YOLOP: host:9777
    
    Args:
        host: Hostname where services are running
        port_base: Base port (4777)
        timeout: Request timeout
    
    Returns:
        Configured PerceptionAggregator instance
    """
    service_config = {
        'cv_lane_detection': f'http://{host}:4777',
        'object_detection': f'http://{host}:5777',
        'traffic_light_detection': f'http://{host}:6777',
        'sign_detection': f'http://{host}:7777',
        'sign_classification': f'http://{host}:8777',
        'yolop': f'http://{host}:9777'
    }
    
    return PerceptionAggregator(service_config, timeout=timeout)