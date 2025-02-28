# Setting Up CARLA 0.9.15 and DonkeySim on Linux Ubuntu VM

This document provides comprehensive instructions to set up CARLA 0.9.15 and DonkeySim in a headless Linux Ubuntu VM using Xvfb. Follow these steps to ensure both simulators are properly configured and can be switched seamlessly for RL training.

---

## 1. System Preparation

Update your package lists and install system dependencies:
```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install xvfb wget tar python3-pip -y
```

*Tip: Consider creating a Python virtual environment for isolation:*
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 2. Installing and Running CARLA 0.9.15

### Download and Extract CARLA

Download the archives (adjust URLs as needed):
```bash
wget https://carla-releases.s3.amazonaws.com/0.9.15/CARLA_0.9.15.tar.gz
wget https://carla-releases.s3.amazonaws.com/0.9.15/AdditionalMaps_0.9.15.tar.gz
tar -xzvf CARLA_0.9.15.tar.gz
tar -xzvf AdditionalMaps_0.9.15.tar.gz
mv CarlaUE4 /home/$USER/carla_0.9.15  # or your preferred location
```

### Install Xvfb

Xvfb provides a virtual display for running graphical apps:
```bash
sudo apt-get install xvfb
```

### Running the CARLA Server

Navigate to the CARLA directory and launch the server:
```bash
cd /home/$USER/carla_0.9.15
xvfb-run -s "-screen 0 1024x768x24" ./CarlaUE4.sh -carla-server -carla-port=2000 -windowed
```

### Installing the CARLA Python API

Set up the CARLA Python API:
```bash
cd /home/$USER/carla_0.9.15/PythonAPI/carla
pip3 install -r requirements.txt
python3 setup.py build
python3 setup.py install
```

---

## 3. Installing and Running DonkeySim

### Install DonkeySim

Install DonkeySim via pip (within your virtual environment if used):
```bash
pip3 install donkeycar[sim]
```

### Running the DonkeySim Server

Launch DonkeySim with Xvfb:
```bash
xvfb-run donkey sim
```

---

## 4. Environment Interface and Switching

To seamlessly switch between CARLA and DonkeySim, create a common environment interface. Both environments should implement methods `reset`, `step`, and `close`. 

### Implementing a DonkeySim Wrapper

Below is an example implementation for DonkeySim (`DonkeyEnv`) that mimics CARLA’s interface:

```python
import donkeycar as dk
import numpy as np
import cv2

class DonkeyEnv:
    def __init__(self, max_steps_per_episode=1000):
        self.donkey = dk.DonkeySim()
        self.state_size = 16
        self.max_steps_per_episode = max_steps_per_episode
        self.step_count = 0
        self.action_space = [(0.5, -0.5), (0.5, 0.0), (0.5, 0.5),
                             (0.0, -0.5), (0.0, 0.0), (0.0, 0.5)]  # Discrete actions

    def process_image(self, img):
        height, width = img.shape[:2]
        sector_size = width // 8
        min_distances = [height] * 8
        max_distances = [0] * 8
        for i in range(8):
            sector = img[:, i*sector_size:(i+1)*sector_size]
            for col in range(sector_size):
                for row in range(height):
                    if sector[row, col, 0] > 0:  # simplified non-sky pixel check
                        min_distances[i] = min(min_distances[i], row)
                        max_distances[i] = max(max_distances[i], row)
                        break
        state = [(min_d / height, max_d / height) for min_d, max_d in zip(min_distances, max_distances)]
        state = [val for pair in state for val in pair]
        return state

    def reset(self):
        self.donkey.reset()
        img = self.donkey.step(0, 0)[0]
        self.step_count = 0
        return self.process_image(img)

    def step(self, action):
        self.step_count += 1
        if isinstance(action, int):
            throttle, steer = self.action_space[action]
        else:
            throttle, steer = action
        img, _, _ = self.donkey.step(throttle, steer)
        state = self.process_image(img)
        reward = 1.0  # Simplified reward; enhance as needed
        done = self.step_count >= self.max_steps_per_episode
        return state, reward, done, {}

    def close(self):
        self.donkey.close()
```

### Defining an Abstract Environment Interface

Implement an abstract class to enforce the interface:
```python
from abc import ABC, abstractmethod

class AbstractEnv(ABC):
    @property
    @abstractmethod
    def state_size(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def close(self):
        pass
```

Ensure your CARLA environment (`CarlaEnv`) inherits from `AbstractEnv` as well.

### Switching Environments During Training

In your training script, select the simulator by setting a flag. For example:
```python
use_carla = True  # Change to False to use DonkeySim

if use_carla:
    env = CarlaEnv()  # Make sure CarlaEnv follows the AbstractEnv interface
else:
    env = DonkeyEnv()

# Training loop remains unchanged.
```

---

## 5. Additional Tips

- Verify that your Python version meets all simulator requirements (Python 3.7+ is recommended).
- Monitor resource usage since Xvfb and graphical simulators might require additional CPU/GPU resources.
- If issues arise with missing libraries for graphical components, consider installing extra dependencies, for example:
```bash
sudo apt-get install libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0
```

---

This comprehensive guide now includes all essential commands and extra details to ensure a smooth setup.

Happy simulating!