import carla
import time

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        client.load_world('Town04')
        world = client.get_world()
        
        carla_map = world.get_map()
        spectator = world.get_spectator()
        
        spawn_points = carla_map.get_spawn_points()
        
        while True:
            spec_transform = spectator.get_transform()
            spec_loc = spec_transform.location
            
            print(f"Spectator: x={spec_loc.x:.2f}, y={spec_loc.y:.2f}, z={spec_loc.z:.2f}")
            
            closest_spawn = None
            min_dist = float('inf')
            
            for spawn in spawn_points:
                dist = spec_loc.distance(spawn.location)
                if dist < min_dist:
                    min_dist = dist
                    closest_spawn = spawn
            
            if closest_spawn:
                c_loc = closest_spawn.location
                print(f" -> Closest Spawn: x={c_loc.x:.2f}, y={c_loc.y:.2f}, z={c_loc.z:.2f} (Dist: {min_dist:.2f}m)")
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("stopped")

if __name__ == '__main__':
    main()