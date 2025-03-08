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

sys.path.append("/teamspace/studios/this_studio")
from LAV_n import CarlaEnvWrapper, PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarlaVisualizer:
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
            self.agent.load_state_dict(torch.load(model_path))
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
            pygame.display.set_caption("CARLA Agent Visualization")
        
        # Setup placeholders for camera and image
        self.camera = None
        self.camera_image = None
        
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
            'lane_deviation': 0.0,
            'collision': False,
            'reward_components': {}
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
        camera_bp.set_attribute('fov', '90')  # Slightly narrower FOV for better focus
        
        # Position camera behind and above the car for proper third-person view
        # Increasing distance and height to ensure vehicle is visible
        camera_transform = carla.Transform(
            carla.Location(x=-6.0, z=3.5, y=0.0),  # 6.0 meters behind, 3.5 meters above
            carla.Rotation(pitch=-20, yaw=0)  # Look down at a steeper angle
        )
        
        # Create the camera and attach it with fixed position relative to vehicle
        self.camera = self.env.world.spawn_actor(camera_bp, camera_transform, attach_to=self.env.vehicle)
        self.camera.listen(lambda image: self.process_camera_image(image))
        logger.info("Third-person camera attached to vehicle")
        
        # Verify the attachment by getting camera location
        camera_location = self.camera.get_location()
        vehicle_location = self.env.vehicle.get_location()
        logger.info(f"Camera position: {camera_location}, Vehicle position: {vehicle_location}")
        
    def process_camera_image(self, image):
        """Process camera image data."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        self.camera_image = array
        
    def update_metrics_display(self, frame):
        """Add metrics overlay to the frame."""
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        
        # Try to use a system font, fallback to default if not available
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
            f"Lane Deviation: {self.metrics['lane_deviation']:.3f}"
        ]
        
        # Add reward components if available
        if self.metrics['reward_components']:
            components_text = []
            for k, v in self.metrics['reward_components'].items():
                components_text.append(f"{k}: {v:.3f}")
            metrics_text.append("Reward Components:")
            metrics_text.extend(components_text)
        
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
            
        return np.array(image)
    
    def start_recording(self, fps=20):
        """Start recording video."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(self.save_path, f"agent_run_{timestamp}.mp4")
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
            
    def run_episode(self, max_steps=1000, record=True):
        """Run a full episode with the loaded agent."""
        if record:
            self.start_recording()
            
        # Reset environment and metrics
        state, _ = self.env.reset()
        
        # Camera needs to be reinitialized after environment reset
        # because the reset destroys and recreates the vehicle
        self.setup_camera()
        
        self.metrics = {
            'reward': 0.0,
            'total_reward': 0.0,
            'steps': 0,
            'distance': 0.0,
            'speed': 0.0,
            'lane_deviation': 0.0,
            'collision': False,
            'reward_components': {}
        }
        
        # Wait for camera to initialize
        time.sleep(1.0)  # Increased wait time to ensure camera is ready
        
        # Take a screenshot at the beginning to verify camera position
        if self.camera_image is not None and not self.headless:
            self.save_screenshot(self.update_metrics_display(self.camera_image))
            logger.info("Initial screenshot saved - verify camera position")
        
        # Add debug visuals to confirm vehicle position
        debug = self.env.world.debug
        prev_pos = None
        
        for step in range(max_steps):
            # Check for pygame events (for window closing)
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
            
            # Get action from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dist, _ = self.agent(state_tensor)
                action = dist.sample().cpu().numpy()[0]
            
            # Take step in environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Update metrics
            self.metrics['reward'] = reward
            self.metrics['total_reward'] += reward
            self.metrics['steps'] += 1
            self.metrics['distance'] = info.get('distance_traveled', 0.0)
            self.metrics['speed'] = info.get('speed_kmh', 0.0)
            self.metrics['lane_deviation'] = info.get('lane_deviation', 0.0)
            self.metrics['collision'] = done  # In our env, done usually means collision
            self.metrics['reward_components'] = info.get('reward_breakdown', {})
            
            # Render frame
            if self.camera_image is not None:
                # Add metrics overlay
                frame = self.update_metrics_display(self.camera_image)
                
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
            
            # Every 10 steps, verify camera is still following the vehicle correctly
            if step % 10 == 0:
                camera_loc = self.camera.get_location()
                vehicle_loc = self.env.vehicle.get_location()
                logger.debug(f"Step {step} - Vehicle at: {vehicle_loc}, Camera at: {camera_loc}")
                
                # If the vehicle seems to be moving but the camera view is static,
                # try teleporting the vehicle to improve visibility
                if step == 20:  # Give some time for initial movement
                    # Save a frame to verify movement
                    if self.camera_image is not None:
                        self.save_screenshot(self.update_metrics_display(self.camera_image))
                        logger.info("Saved frame at step 20 to verify movement")
            
            # Update state
            state = next_state
            
            # Log every 50 steps
            if step % 50 == 0:
                logger.info(f"Step {step}, Reward: {reward:.3f}, Total: {self.metrics['total_reward']:.3f}")
                
            # Check for episode end
            if done or truncated:
                logger.info(f"Episode ended after {step+1} steps with total reward {self.metrics['total_reward']:.3f}")
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
    parser.add_argument("--model", type=str, default="/teamspace/studios/this_studio/manual_logs/best_model.pt",
                        help="Path to the trained model file")
    parser.add_argument("--width", type=int, default=800, help="Visualization width")
    parser.add_argument("--height", type=int, default=600, help="Visualization height")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--no-record", action="store_true", help="Disable video recording")
    
    args = parser.parse_args()
    
    visualizer = CarlaVisualizer(args.model, args.width, args.height, args.headless)
    
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
