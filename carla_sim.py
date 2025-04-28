import carla
import pygame
import numpy as np
import random

# 1. Connect to CARLA and load Town01
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town01')

# 2. Spawn the first vehicle at a random spawn point
blueprint_library = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 3. Create an RGB camera and attach to the vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
# Position: x=1.5m forward, z=2.4m above vehicle's center
camera_transform = carla.Transform(carla.Location(x=-4, z=4), carla.Rotation(pitch=-15))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# 4. Initialize Pygame for display
pygame.init()
display = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Carla Camera")

# 5. Define callback to render camera images
def show_image(image):
    # Convert raw CARLA image to a (H,W,3) NumPy array in RGB order
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    # Create a Pygame surface and blit
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    display.blit(surf, (0, 0))
    pygame.display.flip()

# 6. Start listening to the camera sensor
camera.listen(lambda image: show_image(image))






def detect_lanes(frames):
    return None

def detect_traffic_lights(frames):
    return None

def detect_signs(frames):
    return None


# 7. Keep the script alive until window is closed
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        pygame.time.wait(10)
finally:
    camera.stop()
    vehicle.destroy()
    pygame.quit()