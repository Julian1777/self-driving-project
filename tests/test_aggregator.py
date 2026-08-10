import sys
import numpy as np
import os
import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.communication.aggregator import PerceptionAggregator


def test_health_check():
    service_config = {
        'object_detection': 'http://localhost:5777',
        'traffic_light_detection': 'http://localhost:6777',
        'sign_detection': 'http://localhost:7777',
        'sign_classification': 'http://localhost:8777',
        'yolop': 'http://localhost:9777'
    }
    
    aggregator = PerceptionAggregator(service_config, timeout=2.0)
    try:
        health = aggregator.health_check()
        
        print("Health Check:")
        for service, is_healthy in health.items():
            status = "OK" if is_healthy else "DOWN"
            print(f"  {service}: {status}")
        
        return all(health.values())
    finally:
        aggregator.shutdown()


def test_process_frame():
    service_config = {
        'object_detection': 'http://localhost:5777',
        'traffic_light_detection': 'http://localhost:6777',
        'sign_detection': 'http://localhost:7777',
        'sign_classification': 'http://localhost:8777',
        'yolop': 'http://localhost:9777'
    }
    
    aggregator = PerceptionAggregator(service_config, timeout=10.0)
    test_frame_path = os.path.join(os.path.dirname(__file__), "model_test.jpg")

    if not os.path.exists(test_frame_path):
        print(f"Test image not found: {test_frame_path}")
        return False
    frame = cv2.imread(test_frame_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        result = aggregator.process_frame(
            frame=frame_rgb,
            speed_kph=50.0,
            timestamp_ns=1234567890,
            vehicle_pos=(100.0, 50.0, 0.0),
            vehicle_direction=(1.0, 0.0, 0.0)
        )
        
        print(f"Frame Processing: {result.processing_time_ms:.1f}ms")
        for service, status in result.service_status.items():
            print(f"  {service}: {status}")
        
        # Visualize detections
        output_dir = os.path.join(os.path.dirname(__file__), "detections_output")
        os.makedirs(output_dir, exist_ok=True)
        
        for service_name, service_result in result.results.items():
            if service_result is None:
                continue
            
            vis_frame = frame.copy()
            
            if service_name == 'yolop':
                if 'drivable_area' in service_result and service_result['drivable_area']:
                    da_mask = np.array(service_result['drivable_area'], dtype=np.uint8)
                    if da_mask.ndim == 3:
                        da_mask = da_mask[:, :, 0]
                    da_overlay = cv2.cvtColor(da_mask, cv2.COLOR_GRAY2BGR)
                    da_overlay[:, :] = [0, 255, 0]  # Green color
                    da_overlay[da_mask == 0] = [0, 0, 0]  # Black where mask is 0
                    vis_frame = cv2.addWeighted(vis_frame, 0.7, da_overlay, 0.3, 0)
                
                if 'lane_lines' in service_result and service_result['lane_lines']:
                    ll_mask = np.array(service_result['lane_lines'], dtype=np.uint8)
                    if ll_mask.ndim == 3:
                        ll_mask = ll_mask[:, :, 0]
                    ll_overlay = cv2.cvtColor(ll_mask, cv2.COLOR_GRAY2BGR)
                    ll_overlay[:, :] = [255, 0, 0]  # Blue color
                    ll_overlay[ll_mask == 0] = [0, 0, 0]  # Black where mask is 0
                    vis_frame = cv2.addWeighted(vis_frame, 0.7, ll_overlay, 0.3, 0)
            else:
                if 'detections' in service_result and service_result['detections']:
                    for det in service_result['detections']:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                            conf = det.get('confidence', 0)
                            cls = det.get('class', 'unknown')
                            
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{cls}: {conf:.2f}"
                            cv2.putText(vis_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            output_path = os.path.join(output_dir, f"{service_name}_detections.jpg")
            cv2.imwrite(output_path, vis_frame)
            print(f"  Saved: {output_path}")
        
        return result.all_healthy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        aggregator.shutdown()


def test_error_handling():
    service_config = {
        'object_detection': 'http://localhost:9999',
        'traffic_light_detection': 'http://localhost:6777',
        'sign_detection': 'http://localhost:7777',
        'sign_classification': 'http://localhost:8777',
        'yolop': 'http://localhost:9777'
    }
    
    aggregator = PerceptionAggregator(service_config, timeout=1.0)
    test_frame = np.random.randint(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    
    try:
        result = aggregator.process_frame(
            frame=test_frame,
            speed_kph=50.0,
            timestamp_ns=1234567890
        )
        
        print(f"Partial Results: {result.processing_time_ms:.1f}ms")
        for service, status in result.service_status.items():
            print(f"  {service}: {status}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    finally:
        aggregator.shutdown()


def main():
    print("Running aggregator tests")
    
    tests = [
        ("Health Check", test_health_check),
        ("Process Frame", test_process_frame),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\nInterrupted")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            results[test_name] = False
    
    print("\n\nResults:")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
