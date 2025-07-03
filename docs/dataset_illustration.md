# Dataset Illustration

## Abstract

Our datasets is built on the OpenScene Dataset and further supplemented with carefully designed driving style preferences.

This page will show more clarifications of our Driving Style Label files (styletrain.json & styletest.json).

You can find the download link in [this page](./install.md)

## Data Structure

Each annotated instance in the StyleDrive dataset contains rich semantic, dynamic, and contextual attributes. Below is a sample structure of the data format:

| Variable Name                            | Description                                                                  |
| ---------------------------------------- | ---------------------------------------------------------------------------- |
| vx_ego, vy_ego, v_ego                    | Ego vehicle velocity components (x, y) and magnitude                         |
| ax_ego, ay_ego, a_ego                    | Ego acceleration components and magnitude                                    |
| yaw, yaw_diff                            | Heading angle and its temporal difference                                    |
| v_avg, v_std                             | Average and standard deviation of ego speed (last 10 frames)                 |
| vy_max                                   | Maximum lateral velocity magnitude                                           |
| a_max, a_std, ax_avg                     | Maximum acceleration, std of acceleration, average longitudinal acceleration |
| ini_direction_judge                      | Initial motion direction category                                            |
| scenario_type                            | Scenario type from VLM semantic parsing                                      |
| scene_token                              | Unique scene identifier                                                      |
| has_left_rear / right_rear               | Boolean flags indicating vehicle presence in rear zones                      |
| left_rear_min / right_rear_min           | Minimum relative distance to rear vehicles                                   |
| speed_mode                               | Speed-related preference classification (e.g., aggressive, conservative)     |
| front_frame                              | Distance to front vehicle across frames                                      |
| safe_frame                               | Frame-wise safety flags                                                      |
| safe_ratio, unsafe_ratio, oversafe_ratio | Distribution of safety judgments                                             |
