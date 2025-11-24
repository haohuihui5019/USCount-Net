# USCount-Net

Official implementation of our uncertainty-driven semi-supervised learning framework for apple flower counting with phenological awareness.

Pretrained model evaluation code is available now, full implementation will be released soon.

## 📌 Overview
This repository contains the implementation of a novel semi-supervised learning approach for accurate apple flower counting in agricultural monitoring systems. The method leverages:
- **Adaptive pseudo-label filtering mechanism based on frequent forward uncertainty estimation** to significantly enhances the robustness of semi-supervised flower counting under complex field conditions.
- **Noise-sensitive adaptive gated feature fusion module** to dynamically weights multi-scale features to preserve petal texture details while enhancing flower cluster distribution semantics via a learnable spatial gating mask. This module suppresses noise from redundant features through adaptive fusion.

## 🔒 Code Availability
*We appreciate your understanding in maintaining research integrity during the peer review process. Please watch this repository for updates!*

**Currently released:**  
✅ Pretrained model inference code
✅ Apple flower cluster dataset

**To be released:**  
🔜 Full training implementation  
🔜 Semi-supervised learning framework  

## 📥 Resources
### Pretrained Models
Download pretrained model weights trained on the Flower Cluster dataset with a 30% labeling ratio:
```bash
https://drive.google.com/file/d/1pHtOO1Q6SqisQv0bTXE9oPQ-aoY3MkFc/view?usp=drive_link
```
### Data Set
Prepare the Test data and Train data:
```bash
https://drive.google.com/drive/folders/1KP8H0qIuct56hWre5GV6ZJnzwqOY3Pip?usp=drive_link
```
