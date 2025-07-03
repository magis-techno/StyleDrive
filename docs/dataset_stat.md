# 📊 StyleDrive Dataset Statistics

This page provides key statistics and structure details for the [StyleDrive dataset](https://huggingface.co/datasets/Ryhn98/StyleDrive-Dataset), designed for benchmarking personalized end-to-end autonomous driving.

## Dataset Comparison

| Dataset        | Reality | Scenarios     | CL/OL   | E2E   | Style |
|----------------|---------|---------------|---------|-------|-------|
| nuScenes      | Real    | City          | OL      | ✓     | ✘     |
| OpenScene     | Real    | City & Rural  | OL      | ✓     | ✘     |
| Longest6      | Sim     | City          | CL      | ✓     | ✘     |
| CARLA         | Sim     | City          | CL      | ✓     | ✘     |
| MetaDrive     | Sim     | City          | CL      | ✓     | ✘     |
| Bench2Drive   | Sim     | City          | CL      | ✓     | ✘     |
| NAVSIM        | Real    | City & Rural  | Semi-CL | ✓     | ✘     |
| HuiL-RM       | HuiL    | Ramp-Merge    | OL      | ✘     | ✓     |
| HuiL-CF       | HuiL    | Car-Following | OL      | ✘     | ✓     |
| HuiL-LC       | HuiL    | Lane-Change   | OL      | ✘     | ✓     |
| HuiL-Mul      | HuiL    | City          | OL      | ✘     | ✓     |
| UAH           | Real    | City & Highway| OL      | ✘     | ✓     |
| Brain4Cars    | Real    | Lane-Change & Merge | OL | ✘ | ✓     |
| PDB           | Real    | City          | OL      | ✘     | ✓     |
| **StyleDrive (Ours)** | Real | City & Rural | Semi-CL | ✓     | ✓     |



## 📁 Dataset Overview

| Attribute              | Value                                                  |
|------------------------|--------------------------------------------------------|
| Total Scenarios        | ~29,856                                                |
| ➤ Training Set         | 25,649 scenarios                                       |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• A (Aggressive)   | 3,814                                                  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• N (Normal)       | 20,469                                                 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• C (Conservative) | 1,411                                                  |
| ➤ Testing Set          | 4,096 scenarios                                        |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• A (Aggressive)   | 538                                                    |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• N (Normal)       | 3,275                                                  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• C (Conservative) | 236                                                    |
| Scenario Types         | 11                                                     |
| Annotated Attributes   | 30+ motion, safety, and context fields                 |
| Style Labels           | Aggressive / Normal / Conservative                     |
| Source Dataset         | OpenScene (built upon nuPlan)                         |
| Annotation Method      | Rule-based heuristics + VLM-based inference + Human-in-the-loop |

## Dataset Collection Location (same as nuPlan, but subset)
| Pittsburgh            | Boston               | Las Vegas            | Singapore            |
|-----------------------|----------------------|----------------------|----------------------|
| ![Pittsburgh](/assets/pittsburgh.png) | ![Boston](/assets/boston.png) | ![Las Vegas](/assets/lasvegas.png) | ![Singapore](/assets/singapore.png) |


## 🧭 Traffic Scenario Distribution

| Scenario Type            | Train Count | Test Count | Total Count |
| ------------------------ | ----------- | ---------- | ----------- |
| Lane Following           | 13,385      | 2,345      | 15,730      |
| Lane Change              | 316         | 54         | 370         |
| Protected Intersection   | 3,670       | 582        | 4,252       |
| Unprotected Intersection | 1,191       | 254        | 1,445       |
| Crosswalks               | 1,509       | 241        | 1,750       |
| Side Ego to Main         | 641         | 60         | 701         |
| Side to Main Ego         | 705         | 97         | 802         |
| Countryside Road         | 255         | 97         | 352         |
| Roundabout Entrance      | 115         | 13         | 128         |
| Roundabout Interior      | 30          | 5          | 35          |
| Carpark Areas            | 68          | 39         | 107         |
| Special Interior Road    | 3,809       | 262        | 4,071       |
| **Total**                | **29,856**  | **4,096**  | **33,952**  |

## 🧭 Pie Chart Visualization
![Dataset Overview Image](/assets/alter.png)


## Dataset Detailed Structure and Access Guide

The **StyleDrive dataset** consists of driving scenarios, each described by a rich set of attributes related to vehicle motion, safety, contextual information, and unique scene features. Here is a detailed explanation of the data structure with data types, example data entries, and how to access this data programmatically.

---

## 🧳 **Dataset Structure**

### **Core Features**

Each dataset entry contains the following core features that represent the **ego vehicle's state**, **motion**, **safety**, and **context**. The **data type** for each feature is specified below.

| **Feature**           | **Description**                                                          | **Data Type** | **Example**                  |
| --------------------- | ------------------------------------------------------------------------ | ------------- | ---------------------------- |
| `vx_ego`              | Ego vehicle's velocity along the x-axis (m/s)                            | List\[float]  | `[4.11, 4.14, 4.16, ...]`    |
| `vy_ego`              | Ego vehicle's velocity along the y-axis (m/s)                            | List\[float]  | `[0.01, -0.04, -0.03, ...]`  |
| `v_ego`               | Ego vehicle's speed magnitude (m/s)                                      | List\[float]  | `[4.11, 4.14, 4.17, ...]`    |
| `ax_ego`              | Ego vehicle's acceleration along the x-axis (m/s²)                       | List\[float]  | `[0.04, 0.08, 0.04, ...]`    |
| `ay_ego`              | Ego vehicle's acceleration along the y-axis (m/s²)                       | List\[float]  | `[-0.65, -0.64, -1.18, ...]` |
| `a_ego`               | Ego vehicle's overall acceleration (m/s²)                                | List\[float]  | `[0.65, 0.65, 1.18, ...]`    |
| `yaw`                 | Ego vehicle's yaw angle (radians)                                        | List\[float]  | `[2.16, 2.09, 1.98, ...]`    |
| `yaw_diff`            | Change in yaw angle between frames (radians)                             | List\[float]  | `[-0.12, -0.22, -0.31, ...]` |
| `min_front_frame`     | Minimum distance to the front vehicle (meters)                           | List\[float]  | `[9, 22.69]`                 |
| `v_avg`               | Average speed of ego vehicle over the last 10 frames (m/s)               | float         | `4.76`                       |
| `v_std`               | Standard deviation of speed over the last 10 frames (m/s)                | float         | `0.67`                       |
| `vy_max`              | Maximum lateral velocity in the last 10 frames (m/s)                     | float         | `0.13`                       |
| `a_max`               | Maximum acceleration in the last 10 frames (m/s²)                        | float         | `2.34`                       |
| `ax_avg`              | Average longitudinal acceleration over the last 10 frames (m/s²)         | float         | `0.46`                       |
| `a_std`               | Standard deviation of acceleration over the last 10 frames (m/s²)        | float         | `0.59`                       |
| `ini_direction_judge` | Initial direction of the ego vehicle (e.g., "forward", "backward")       | string        | `"right"`                    |
| `scenario_type`       | Type of driving scenario (e.g., "lane\_following", "crosswalks")         | string        | `"protected_intersections"`  |
| `scene_token`         | Unique identifier for each scene                                         | string        | `"a631fec170525388"`         |
| `has_left_rear`       | Boolean indicating presence of vehicle in the left rear zone             | bool          | `false`                      |
| `has_right_rear`      | Boolean indicating presence of vehicle in the right rear zone            | bool          | `true`                       |
| `left_rear_min`       | Minimum distance to the vehicle in the left rear zone (meters)           | float or null | `null`                       |
| `right_rear_min`      | Minimum distance to the vehicle in the right rear zone (meters)          | float         | `9.69`                       |
| `speed_mode`          | Speed-related preference label (Aggressive, Normal, Conservative)        | string        | `"accelerating"`             |
| `front_frame`         | Temporal distances to the front vehicle across multiple frames (meters)  | List\[float]  | `[-1, -1, -1, -1, 22.69]`    |
| `safe_frame`          | Safety flags for each frame (indicates safe or unsafe driving condition) | List\[bool]   | `[-1, -1, -1, -1, 1]`        |
| `safe_ratio`          | Proportion of frames labeled as "safe"                                   | float         | `1.0`                        |
| `unsafe_ratio`        | Proportion of frames labeled as "unsafe"                                 | float         | `0.0`                        |
| `oversafe_ratio`      | Proportion of frames with excessive safety margin                        | float         | `0.0`                        |

---

### 🧑‍💻 Example Dataset Entry

#### **Example Data** (For Scenario: `protected_intersections`)

```json
{
  "vx_ego": [4.11, 4.14, 4.17, 4.21, 4.35, 4.60, 4.95, 5.31, 5.66, 6.08],
  "vy_ego": [0.01, -0.04, -0.03, -0.04, -0.01, 0.04, 0.07, 0.13, 0.13, 0.13],
  "v_ego": [4.11, 4.14, 4.17, 4.21, 4.35, 4.60, 4.95, 5.31, 5.66, 6.08],
  "ax_ego": [0.04, 0.08, 0.04, 0.13, 0.41, 0.65, 0.80, 0.75, 0.73, 0.99],
  "ay_ego": [-0.65, -0.64, -1.18, -1.43, -1.58, -1.85, -2.17, -2.22, -1.79, -1.67],
  "a_ego": [0.65, 0.65, 1.18, 1.43, 1.63, 1.96, 2.31, 2.34, 1.94, 1.94],
  "yaw": [2.16, 2.09, 1.98, 1.83, 1.65, 1.44, 1.23, 1.00, 0.82, 0.67],
  "yaw_diff": [-0.13, -0.22, -0.31, -0.36, -0.40, -0.43, -0.45, -0.36, -0.29, -0.29],
  "min_front_frame": [9, 22.69],
  "v_avg": 4.76,
  "v_std": 0.67,
  "vy_max": 0.13,
  "a_max": 2.34,
  "ax_avg": 0.46,
  "a_std": 0.59,
  "ini_direction_judge": "right",
  "scenario_type": "protected_intersections",
  "scene_token": "a631fec170525388",
  "has_left_rear": false,
  "has_right_rear": true,
  "left_rear_min": null,
  "right_rear_min": 9.69,
  "speed_mode": "accelerating",
  "front_frame": [-1, -1, -1, -1, 22.69],
  "safe_frame": [-1, -1, -1, -1, 1],
  "safe_ratio": 1.0,
  "unsafe_ratio": 0.0,
  "oversafe_ratio": 0.0,
  "ANC_result": "N",
  "scene_scenario": "protected_intersections",
  "lane_change_frame": -1,
  "token": "a631fec170525388",
  "pedestrians": "No",
  "with_lead": "Far"
}
```

## 🧑‍💻 Example Code to Access Data

Here’s an example in Python to access the dataset attributes:

```python
import json

# Load dataset from a JSON file (assuming the file is in your working directory)
with open('path_to_file.json', 'r') as f:
    data = json.load(f)

# Access specific scene (for example, scene with token 'a631fec170525388')
scene_data = data['a631fec170525388']

# Print velocity data of the ego vehicle
print("Ego Vehicle Velocity (vx_ego):", scene_data['vx_ego'])

# Access the scenario type
print("Scenario Type:", scene_data['scenario_type'])

# Calculate the average speed (v_ego)
v_avg = sum(scene_data['v_ego']) / len(scene_data['v_ego'])
print("Average Speed:", v_avg)

# Check if there is a lead vehicle in the scene
if scene_data['with_lead'] != 'No':
    print("Lead vehicle present:", scene_data['with_lead'])
```

## 🧑‍💻 **Accessing Scene-Specific Features**

For scenario-specific features like **`pedestrians`** or **`merge_risk`**, you can directly access them as:

```python
# Example: Access pedestrians feature for a scenario
print("Pedestrians present:", scene_data.get('pedestrians', 'Unknown'))

# Example: Access merge_risk for a scenario
print("Merge risk level:", scene_data.get('merge_risk', 'None'))
```