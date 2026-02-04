# Radar-Based Multiclass Object Classifier

A machine learning project for classifying objects detected by 2D radar systems in railway environments (CAF I+D). The analysis compares regularized logistic regression (GLMNet) with gradient boosting (XGBoost) for multiclass classification using radar detection features.

![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Railway_Safety-red.svg)

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Object Classes](#object-classes)
- [Methods](#methods)
- [Key Findings](#key-findings)
- [Results](#results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Author](#author)

## Overview

This project develops a multiclass classifier for objects detected by 2D radar systems mounted on trains. The classifier distinguishes between vehicles, pedestrians, cyclists, and static background based on radar detection features such as range, azimuth angle, radar cross-section (RCS), and radial velocity.

**Business Context:** Accurate object classification is critical for railway safety systems, enabling autonomous trains to detect and respond appropriately to obstacles on or near the tracks.

## Dataset

400,000+ radar detections with the following features:

| Variable | Description | Units |
|----------|-------------|-------|
| `range_sc` | Radial distance to detection (sensor coordinates) | meters |
| `azimuth_sc` | Azimuth angle to detection (sensor coordinates) | radians |
| `radar_cross_section` | Radar Cross Section of detection | dBsm |
| `radial_velocity` | Measured radial velocity | m/s |
| `vr_compensated` | Radial velocity compensated for ego-vehicle motion | m/s |
| `x_cc`, `y_cc` | Detection position (ego-vehicle coordinates) | meters |
| `x_seq`, `y_seq` | Detection position (global coordinates) | meters |

### Radar Detection Interface (RDI)
Each detection provides range, radial velocity, azimuth angle, RCS, and probability for multiple reflection points per object.

## Object Classes

| Code | Class | Description |
|------|-------|-------------|
| 0 | Car | Standard passenger vehicles |
| 1 | Large Vehicle | Trucks, buses (merged from codes 2, 3) |
| 5 | Bicycle | Cyclists |
| 7 | Person | Pedestrians (includes groups, code 8) |
| 10 | Other Dynamic | Other moving objects |
| 11 | Static Background | Fixed infrastructure/environment |

**Class Merging:**
- Truck (2) + Bus (3) → Large Vehicle (1)
- Person Group (8) → Person (7)

## Methods

### Data Preprocessing
- Robust outlier removal using MAD-based z-scores (threshold: 6 MADs)
- Class label reassignment for merged categories
- Train/Validation split: 200k/200k records

### Statistical Analysis
- RCS comparison between Person and Large Vehicle classes
- Shapiro-Wilk normality tests
- Wilcoxon rank-sum test with Cliff's Delta effect size
- ROC/AUC analysis for RCS discriminatory capacity

### Classification Models

#### 1. GLMNet (Multinomial Logistic Regression)
- Elastic net regularization (α = 0.5)
- Feature interactions: RCS×range, RCS×azimuth
- 5-fold cross-validation for λ selection

#### 2. XGBoost (Gradient Boosting)
- Multi-class soft probability output
- Early stopping (20 rounds patience)
- Hyperparameters: η=0.1, max_depth=6, subsample=0.8

### Evaluation Metrics
- **AP (Average Precision):** Per-class, One-vs-Rest
- **mAP (mean Average Precision):** Overall classifier performance
- **AUC-ROC:** Discriminatory capacity per class
- **Precision-Recall curves:** Operating point selection

## Key Findings

### 1. RCS Discriminates Between Person and Large Vehicle
- Wilcoxon test: p ≈ 0 (highly significant)
- Cliff's Delta: -0.548 (large effect)
- AUC: 0.774 → Very good discriminatory capacity
- Large vehicles consistently show higher RCS values across distances

### 2. XGBoost Outperforms GLMNet

| Metric | GLMNet | XGBoost |
|--------|--------|---------|
| mAP (Train) | 0.269 | 0.933 |
| mAP (Validation) | 0.269 | 0.522 |

XGBoost improves validation mAP by +0.253, demonstrating its ability to capture non-linear relationships.

### 3. Per-Class Performance (XGBoost Validation)

| Class | AP | AUC |
|-------|-----|-----|
| Person (7) | 0.850 | 0.952 |
| Car (0) | 0.540 | 0.912 |
| Static Background (11) | 0.772 | 0.912 |
| Large Vehicle (1) | 0.355 | 0.790 |
| Other Dynamic (10) | 0.304 | 0.767 |
| Bicycle (5) | 0.271 | 0.605 |

### 4. Distance and Azimuth Effects
- **Cars:** Best AP at short distances (~10m), degrades beyond 15m
- **Persons:** Surprisingly, AP improves with distance (stabilizes at ~0.85)
- **Bicycles:** AP decreases monotonically with distance
- **Azimuth:** Most classes perform better near boresight (|azimuth| < 0.1 rad)

### 5. Operating Point Selection (Person Class)
- **Priority:** High Precision (≥95%)
- **Selected Threshold:** 0.293
- **Train:** Precision=0.951, Recall=0.999
- **Validation:** Precision=0.827, Recall=0.973

## Results

```
┌─────────────────────────────────────────────────────────────┐
│              XGBOOST MULTICLASS CLASSIFIER                  │
├─────────────────────────────────────────────────────────────┤
│  mAP (Train): 0.933  |  mAP (Validation): 0.522             │
│  Best iteration: 200 rounds with early stopping             │
│  Most important feature: vr_compensated                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              PERSON DETECTION (Class 7)                     │
├─────────────────────────────────────────────────────────────┤
│  Operating Point: Threshold = 0.293                         │
│  Train:      Precision = 0.951 | Recall = 0.999             │
│  Validation: Precision = 0.827 | Recall = 0.973             │
│  High recall maintained; precision drops moderately         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              KEY INSIGHTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • RCS is a strong discriminator (AUC=0.774)                │
│  • vr_compensated is the most predictive feature            │
│  • Bicycle classification remains challenging (AUC=0.605)   │
│  • Model shows overfitting (mAP train >> mAP val)           │
│  • Azimuth affects detection quality as expected            │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
radar-object-classifier/
│
├── README.md                    # Project documentation
├── ASSIGNMENT.md               # Original assignment description
│
├── src/
│   └── radar_multiclass_classifier.R   # Main analysis script
│
└── data/
    └── data.RData       # Dataset (not included)
```