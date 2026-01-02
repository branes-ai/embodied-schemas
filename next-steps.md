
  Data Expansion Options:

  1. More Hardware Platforms - Add other Jetson variants (Orin NX, AGX Orin), Raspberry Pi 5, Hailo-8, etc.
  2. ARM Server CPUs - Ampere Altra, AWS Graviton 3/4, NVIDIA Grace - these are increasingly important for AI inference
  3. Link Jetson GPUs - Create GPUEntry for Orin's integrated GPU and link via gpu_id field (complete the cross-reference we designed)
  4. Seed Models/Sensors/UseCases - The data directories exist but are empty - could add YOLOv8, depth models, camera specs, drone/robot use case templates

  Integration Options:

  5. graphs repo sync - Ensure the graphs repo can consume these schemas, check if any adapter work is needed
  6. Embodied-AI-Architect integration - Update that repo to use the new CPU/GPU data in its tools

  Schema Refinement:

  7. CPU Architecture Summaries - Create CPUArchitectureSummary entries (like we have for GPUs) describing Zen 4, Zen 5, Raptor Lake, etc.
  8. NPU/Accelerator Schema - Add schemas for dedicated AI accelerators (Hailo-8, Coral TPU, Intel NPU)




● All Data Expansion Options Complete!

  | Option                          | Status | Summary                                                       |
  |---------------------------------|--------|---------------------------------------------------------------|
  | 1. More Hardware Platforms      | ✅     | 11 devices (Jetson Orin/Thor/TX2, Pi 4/5, Hailo-8/8L/10H)     |
  | 2. ARM Server CPUs              | ✅     | 11 CPUs (Ampere, Graviton, Grace, NXP, ST Micro, mobile SoCs) |
  | 3. Link Jetson GPUs             | ✅     | 6 embedded GPUs with cross-references                         |
  | 4. Seed Models/Sensors/UseCases | ✅     | Just completed:                                               |

  Seed Data Added:

  | Category              | Count | Entries                                  |
  |-----------------------|-------|------------------------------------------|
  | Models - Detection    | 5     | YOLOv8n/s/m/l, RT-DETR-L                 |
  | Models - Depth        | 2     | MiDaS Small, Depth Anything V2 Small     |
  | Models - Segmentation | 2     | YOLOv8n-seg, SAM ViT-B                   |
  | Models - Pose         | 1     | RTMPose-M                                |
  | Sensors - Cameras     | 1     | Arducam IMX477                           |
  | Sensors - Depth       | 2     | Intel RealSense D435i, Luxonis OAK-D Pro |
  | Sensors - LiDAR       | 1     | Livox Mid-360                            |
  | Use Cases - Drone     | 2     | Obstacle avoidance, Visual inspection    |
  | Use Cases - Quadruped | 1     | Security patrol                          |
  | Use Cases - AMR       | 1     | Warehouse navigation                     |

