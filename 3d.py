import open3d as o3d

pcd = o3d.io.read_point_cloud("aloe.ply")
print(pcd)
print(pcd.get_center().round(2))
print(pcd.get_axis_aligned_bounding_box()
      .get_extent().round(2))
small = pcd.voxel_down_sample(voxel_size=0.01)

clean, _ = small.remove_statistical_outlier(
    nb_neighbors=200, std_ratio=5.0)
print(len(pcd.points), "->", len(small.points),
      "->", len(clean.points))
o3d.visualization.draw_geometries([clean])