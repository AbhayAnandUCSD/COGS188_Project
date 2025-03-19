import os
import sys
import time
import numpy as np
import torch
import carla
import pygame
import argparse
import logging
from datetime import datetime
import cv2
from PIL import Image, ImageDraw, ImageFont

# Import from our training script instead of LAV_n
sys.path.append("/teamspace/studios/this_studio")
from train_ppo_carla import CarlaEnvWrapper, PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PPOVisualizer:
    def __init__(self, model_path, width=800, height=600, headless=False):
        """Initialize the visualizer with the trained model and render settings."""
        self.width = width
        self.height = height
        self.headless = headless
        self.model_path = model_path
        self.save_path = os.path.join("/teamspace/studios/this_studio", "visualizations")
        os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize environment
        self.env = CarlaEnvWrapper()
        
        # Initialize agent and load model
        self.agent = PPOAgent(env=self.env)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            # Support both direct state_dict and Lightning checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.agent.load_state_dict(checkpoint['state_dict'])
            else:
                self.agent.load_state_dict(checkpoint)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file {model_path} not found, using untrained agent")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)
        self.agent.eval()  # Set to evaluation mode
        
        # Initialize pygame for visualization
        if not self.headless:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("PPO Agent Visualization")
        
        # Setup placeholders for camera and image
        self.camera = None
        self.camera_image = None
        self.lidar_display = None
        
        # Reset environment first to create the vehicle
        logger.info("Resetting environment to initialize vehicle...")
        self.env.reset()
        
        # Now setup the camera - after vehicle is created by reset()
        self.setup_camera()
        
        # Metrics tracking
        self.metrics = {
            'reward': 0.0,
            'total_reward': 0.0,
            'steps': 0,
            'distance': 0.0,
            'speed': 0.0,
            'lane_invasions': 0,
            'collision': False
        }
        
        # Video recording setup
        self.record_video = False
        self.video_writer = None
        self.frame_buffer = []
        
    def setup_camera(self):
        """Setup a third-person camera to follow the vehicle."""
        if self.camera:
            self.camera.destroy()
        
        camera_bp = self.env.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        camera_bp.set_attribute('fov', '90')
        
        # Position camera behind and above the car
        camera_transform = carla.Transform(
            carla.Location(x=-6.0, z=3.5, y=0.0),
            carla.Rotation(pitch=-20, yaw=0)
        )
        
        # Create the camera and attach it with fixed position relative to vehicle
        self.camera = self.env.world.spawn_actor(camera_bp, camera_transform, attach_to=self.env.vehicle)
        self.camera.listen(lambda image: self.process_camera_image(image))
        logger.info("Third-person camera attached to vehicle")
        
    def process_camera_image(self, image):
        """Process camera image data."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        self.camera_image = array

    def create_lidar_visualization(self, state):
        """Create a visualization of the LiDAR state data."""
        # Create a top-down view visualization of the LiDAR sectors
        vis_size = 200
        vis_image = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
        center = (vis_size // 2, vis_size // 2)
        
        # Draw background circle
        cv2.circle(vis_image, center, vis_size // 2 - 2, (20, 20, 20), -1)
        cv2.circle(vis_image, center, vis_size // 2 - 2, (40, 40, 40), 1)
        
        # Draw vehicle
        cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
        
        # Draw LiDAR sectors
        for i in range(8):
            angle_start = -180 + i * 45
            angle_end = -180 + (i+1) * 45
            mid_angle = (angle_start + angle_end) / 2 * np.pi / 180
            
            # Get min and max distances from state
            min_dist = state[i*2]
            max_dist = state[i*2 + 1]
            
            # Scale distances for visualization
            scale_factor = (vis_size // 2 - 10) / 50.0  # 50 is max LiDAR range
            min_dist_scaled = min(min_dist * scale_factor, vis_size // 2 - 10)
            
            # Calculate endpoint for visualization
            end_x = center[0] + int(min_dist_scaled * np.cos(mid_angle))
            end_y = center[1] + int(min_dist_scaled * np.sin(mid_angle))
            
            # Draw line for minimum distance in this sector
            color = (0, 255, 0) if min_dist > 5.0 else (0, 165, 255)  # Green if clear, orange if close
            cv2.line(vis_image, center, (end_x, end_y), color, 2)
            
        return vis_image
        
    def update_metrics_display(self, frame, state=None):
        """Add metrics overlay to the frame."""
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Add text for metrics
        y_offset = 10
        text_color = (255, 255, 255)
        shadow_color = (0, 0, 0)
        
        metrics_text = [
            f"Steps: {self.metrics['steps']}",
            f"Total Reward: {self.metrics['total_reward']:.2f}",
            f"Current Reward: {self.metrics['reward']:.2f}",
            f"Distance: {self.metrics['distance']:.2f} m",
            f"Speed: {self.metrics['speed']:.1f} km/h",
            f"Lane Invasions: {self.metrics['lane_invasions']}"
        ]
        
        # Add collision indicator
        if self.metrics['collision']:
            metrics_text.append("COLLISION DETECTED!")
            
        # Draw text with shadow for better visibility
        for i, text in enumerate(metrics_text):
            position = (10, y_offset + i * 20)
            # Draw shadow
            draw.text((position[0]+1, position[1]+1), text, font=font, fill=shadow_color)
            # Draw text
            draw.text(position, text, font=font, fill=text_color)
        
        # Add LiDAR visualization if state is provided
        if state is not None:
            lidar_vis = self.create_lidar_visualization(state)
            lidar_vis_pil = Image.fromarray(lidar_vis)
            # Position in top-right corner
            image.paste(lidar_vis_pil, (self.width - lidar_vis.shape[1] - 10, 10))
            
        return np.array(image)
    
    def start_recording(self, fps=20):
        """Start recording video."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(self.save_path, f"ppo_run_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (self.width, self.height))
        self.record_video = True
        self.frame_buffer = []
        logger.info(f"Started recording to {video_path}")
        
    def stop_recording(self):
        """Stop recording and save video."""
        if self.record_video and self.video_writer:
            # Write any buffered frames
            for frame in self.frame_buffer:
                self.video_writer.write(frame)
            
            self.video_writer.release()
            self.record_video = False
            self.frame_buffer = []
            logger.info("Video recording completed")
            
    def save_screenshot(self, frame):
        """Save a screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        screenshot_path = os.path.join(self.save_path, f"screenshot_{timestamp}.png")
        cv2.imwrite(screenshot_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        logger.info(f"Screenshot saved to {screenshot_path}")
        
    def visualize_action(self, action, frame):
        """Visualize the current action (throttle/brake and steering)."""
        h, w = frame.shape[:2]
        throttle_brake, steering = action
        
        # Create control visualization
        control_vis = np.ones((100, 200, 3), dtype=np.uint8) * 60  # Dark gray background
        
        # Draw throttle/brake bar
        throttle_color = (0, 255, 0)  # Green for throttle
        brake_color = (0, 0, 255)     # Red for brake
        
        bar_height = 20
        bar_width = 150
        bar_x = 25
        throttle_y = 30
        
        # Draw background bar
        cv2.rectangle(control_vis, (bar_x, throttle_y), (bar_x + bar_width, throttle_y + bar_height), 
                     (120, 120, 120), -1)
        
        # Draw throttle/brake level
        if throttle_brake > 0:  # Throttle
            level_width = int(throttle_brake * bar_width)
            cv2.rectangle(control_vis, (bar_x, throttle_y), 
                         (bar_x + level_width, throttle_y + bar_height), 
                         throttle_color, -1)
        else:  # Brake
            level_width = int(-throttle_brake * bar_width)
            cv2.rectangle(control_vis, (bar_x, throttle_y), 
                         (bar_x + level_width, throttle_y + bar_height), 
                         brake_color, -1)
        
        # Add labels
        cv2.putText(control_vis, "Throttle/Brake", (bar_x, throttle_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw steering wheel
        steering_y = 70
        wheel_center = (bar_x + bar_width // 2, steering_y)
        wheel_radius = 20
        
        # Draw wheel background
        cv2.circle(control_vis, wheel_center, wheel_radius, (120, 120, 120), -1)
        cv2.circle(control_vis, wheel_center, wheel_radius, (200, 200, 200), 1)
        
        # Draw steering position
        angle = -steering * np.pi  # Convert [-1, 1] to [-pi, pi]
        end_x = int(wheel_center[0] + wheel_radius * np.cos(angle))
        end_y = int(wheel_center[1] + wheel_radius * np.sin(angle))
        cv2.line(control_vis, wheel_center, (end_x, end_y), (0, 255, 255), 2)
        
        # Add label
        cv2.putText(control_vis, "Steering", (bar_x, steering_y - wheel_radius - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Overlay control visualization in the bottom-right corner of the frame
        frame_with_controls = frame.copy()
        corner_y = h - control_vis.shape[0] - 10
        corner_x = w - control_vis.shape[1] - 10
        frame_with_controls[corner_y:corner_y+control_vis.shape[0], 
                           corner_x:corner_x+control_vis.shape[1]] = control_vis
        
        return frame_with_controls
            
    def run_episode(self, max_steps=1000, record=True):
        """Run a full episode with the loaded agent."""
        if record:
            self.start_recording()
            
        # Reset environment and metrics
        state, _ = self.env.reset()
        
        # Camera needs to be reinitialized after environment reset
        self.setup_camera()
        
        self.metrics = {
            'reward': 0.0,
            'total_reward': 0.0,
            'steps': 0,
            'distance': 0.0,
            'speed': 0.0,
            'lane_invasions': 0,
            'collision': False
        }
        
        # Wait for camera to initialize
        time.sleep(1.0)
        
        # Take a screenshot at the beginning to verify camera position
        if self.camera_image is not None and not self.headless:
            self.save_screenshot(self.update_metrics_display(self.camera_image, state))
            logger.info("Initial screenshot saved - verify camera position")
        
        # Add debug visuals to confirm vehicle position
        debug = self.env.world.debug
        prev_pos = None
        
        for step in range(max_steps):
            # Check for pygame events
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:  # Save screenshot with 's' key
                            if self.camera_image is not None:
                                self.save_screenshot(self.update_metrics_display(self.camera_image, state))
            
            # Get action from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dist, value = self.agent(state_tensor)
                action = dist.sample().cpu().numpy()[0]
                value = value.item()  # Get the critic's value estimate
            
            # Take step in environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Update metrics
            self.metrics['reward'] = reward
            self.metrics['total_reward'] += reward
            self.metrics['steps'] += 1
            self.metrics['distance'] = info.get('distance_traveled', 0.0)
            
            # Get vehicle speed in km/h
            velocity = self.env.vehicle.get_velocity()
            speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            self.metrics['speed'] = speed_kmh
            
            self.metrics['lane_invasions'] = info.get('lane_invasions', 0)
            self.metrics['collision'] = done  # In our env, done usually means collision
            
            # Render frame
            if self.camera_image is not None:
                # Add metrics overlay and visualize actions
                frame = self.update_metrics_display(self.camera_image, state)
                frame = self.visualize_action(action, frame)
                
                # Display frame
                if not self.headless:
                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    self.display.blit(surface, (0, 0))
                    pygame.display.flip()
                
                # Record frame if required
                if self.record_video:
                    self.frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    # Periodically write frames to avoid memory issues
                    if len(self.frame_buffer) >= 100:
                        for f in self.frame_buffer:
                            self.video_writer.write(f)
                        self.frame_buffer = []
            
            # Add debug visualization to track vehicle movement
            current_pos = self.env.vehicle.get_location()
            if prev_pos:
                # Draw a line showing the vehicle's path
                debug.draw_line(
                    prev_pos,
                    current_pos,
                    thickness=0.1,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=10.0
                )
            prev_pos = current_pos
            
            # Draw the next waypoints if available
            if self.env.next_waypoint:
                debug.draw_point(
                    self.env.next_waypoint.transform.location + carla.Location(z=0.5),
                    size=0.1,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=0.1
                )
            
            # Update state
            state = next_state
            
            # Log every 50 steps
            if step % 50 == 0:
                logger.info(f"Step {step}, Reward: {reward:.3f}, Total: {self.metrics['total_reward']:.3f}, Value: {value:.3f}")
                
            # Check for episode end
            if done or truncated:
                logger.info(f"Episode ended after {step+1} steps with total reward {self.metrics['total_reward']:.3f}")
                # Save final screenshot
                if self.camera_image is not None and not self.headless:
                    self.save_screenshot(self.update_metrics_display(self.camera_image, state))
                break
                
        # End recording
        if record:
            self.stop_recording()
    
    def cleanup(self):
        """Clean up resources."""
        if self.record_video:
            self.stop_recording()
        
        if self.camera:
            self.camera.destroy()
            
        self.env.close()
        
        if not self.headless:
            pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Visualize a trained PPO agent in CARLA")
    parser.add_argument("--model", type=str, default="/teamspace/studios/this_studio/logs/ppo_checkpoint_epoch=003.ckpt",
                        help="Path to the trained model file")
    parser.add_argument("--width", type=int, default=800, help="Visualization width")
    parser.add_argument("--height", type=int, default=600, help="Visualization height")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--no-record", action="store_true", help="Disable video recording")
    
    args = parser.parse_args()
    
    visualizer = PPOVisualizer(args.model, args.width, args.height, args.headless)
    
    try:
        for episode in range(args.episodes):
            logger.info(f"Running episode {episode+1}/{args.episodes}")
            visualizer.run_episode(max_steps=args.max_steps, record=not args.no_record)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        visualizer.cleanup()

if __name__ == "__main__":
    main()
