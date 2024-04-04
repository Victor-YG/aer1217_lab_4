# Lab 4 Documentation

## Code Structure
The main Visual Odometry (VO) code is implemented in the `pose_estimation()` function. For simplicity, the code is split into modular subfunctions for each task: `reprojection()` to transform matched feature points to 3D world coordinates, `ransac()` to filter outlier matches, and `point_cloud_alignment()` to perform point cloud alignment with the filtered features and return the corresponding rotation matrix and translation vector.

## To Run
Ensure the `unzip` utility is installed (or manually unzip the .zip archive). Extract the KITTI image dataset and the ground truth pose information into the same directory as the lab code.

```bash
cd <path_to_lab4_code>
unzip CityData.zip -d <path_to_lab4_code>
cp ground_truth_pose.mat <path_to_lab4_code>
python3 lab4.py
```

