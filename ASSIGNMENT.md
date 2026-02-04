# Assignment: Multiclass Object Classifier Based on 2D Radar

## Objective

Generation of a multiclass object classifier based on 2D radar data.


## Background Information

### Radar Detection Interface (RDI)
The radar system provides multiple detections per scan, each with range, radial velocity, azimuth angle, RCS (Radar Cross Section), and probability values.

### Response Variable (y): Detected Object Class

| Code | Class |
|------|-------|
| 0 | Car |
| 1 | Large Vehicle |
| 2 | Truck |
| 3 | Bus |
| 5 | Bicycle |
| 7 | Person |
| 8 | Group of People |
| 10 | Other Dynamic Object |
| 11 | Static Background |

### Explanatory Variables (X)

| Variable | Description |
|----------|-------------|
| `range_sc` | Radial distance [m] to the detection in sensor-relative coordinates |
| `azimuth_sc` | Azimuth angle [radians] to the detection in sensor-relative coordinates |
| `radar_cross_section` | Radar Cross Section [dBsm] of the detection |
| `radial_velocity` | Radial velocity [m/s] measured for this detection |
| `vr_compensated` | Radial velocity [m/s] compensated for ego-vehicle motion |
| `x_cc` | X position [m] in ego-vehicle relative coordinates |
| `y_cc` | Y position [m] in ego-vehicle relative coordinates |
| `x_seq` | X position [m] in global coordinates |
| `y_seq` | Y position [m] in global coordinates |

### Important Notes
- Assume each detection (record) as an independent object (problem simplification).
- As we work with larger absolute azimuth angles, reliability decreases.

---

## Tasks

### 1) Load the data "data.RData" 

### 2) Reassign class labels 
- Reassign bus (3) and truck (2) labels to large vehicle (1)
- Reassign group of people (8) label to person (7) in the response variable y

### 3) Split initial data (X and y) into "train" and "validation" 
- **Train:** First 200k records
- **Validation:** Next 200k records

### 4) Plot median ± MAD of Radar Cross Section (RCS) for Person (7) and Large Vehicle (1) classes as a function of distance (range) in the Train set 
- Is there a significant difference in RCS between Person (7) and Large Vehicle (1) classes?
- Justify your answer.

### 5) Generate a multiclass object model and obtain AP (Average Precision) per class vs rest and mAP (mean Average Precision) in training and validation

**NOTE 1:** It is recommended to add interactions between explanatory variables.

**NOTE 2:** With computational resource limitations, it is recommended to sample from the training set.

### 6) Do AP values per class vs rest improve at closer distances and azimuth angles closer to 0? 
- Justify your answer.

### 7) Choose an operating point (threshold) from the PR curve associated with the Person class vs rest in training 
- Verify the Precision and Recall values in validation corresponding to the threshold previously defined in training.
- **Priority:** Model precision should be prioritized.
- Are the results similar between training and validation?
- Justify your answer.