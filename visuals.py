import math
import matplotlib.pyplot as plt
import numpy as np
import json

with open("punk/small/comparison_0.json") as json_file:
    data = json.load(json_file)


# original
original_linear_blue = data["original"]["linrgb"]["blue"]
original_linear_blue_data = np.array(original_linear_blue["data"])

original_linear_green = data["original"]["linrgb"]["green"]
original_linear_green_data = np.array(original_linear_green["data"])

original_linear_red = data["original"]["linrgb"]["red"]
original_linear_red_data = np.array(original_linear_red["data"])
print(len(original_linear_blue_data), len(original_linear_green_data), len(original_linear_red_data))
# copy
copy_linear_blue = data["copy"]["linrgb"]["blue"]
copy_linear_blue_data = np.array(copy_linear_blue["data"])

copy_linear_green = data["copy"]["linrgb"]["green"]
copy_linear_green_data = np.array(copy_linear_green["data"])

copy_linear_red = data["copy"]["linrgb"]["red"]
copy_linear_red_data = np.array(copy_linear_red["data"])
print(len(copy_linear_blue_data), len(copy_linear_green_data), len(copy_linear_red_data))


fig = plt.figure(figsize=(15, 15), dpi=100)
rgb = fig.add_subplot(projection='3d')
# rgb.view_init(30, 300)
# rgb.view_init(15, 500)
rgb.view_init(-5, 595)
choice = np.random.choice(len(original_linear_blue_data), 8000, replace=False)

copy_fig = plt.figure(figsize=(15, 15), dpi=100)
copy_rgb = copy_fig.add_subplot(projection='3d')
# copy_rgb.view_init(30, 300)
# copy_rgb.view_init(15, 500)
copy_rgb.view_init(-5, 595)


rgb.scatter(
    xs=original_linear_red_data[choice],
    ys=original_linear_green_data[choice],
    zs=original_linear_blue_data[choice],
    marker=".",
    s=1
)

copy_rgb.scatter(
    xs=copy_linear_red_data,
    ys=copy_linear_green_data,
    zs=copy_linear_blue_data,
    marker=".",
    s=1
)

rgb.set_title("Orig RGB")
rgb.set_xlabel('Red')
rgb.set_ylabel('Green')
rgb.set_zlabel('Blue')
# rgb.set_xlim(0, 1)
# rgb.set_ylim(0, 1)
# rgb.set_zlim(0, 1)


copy_rgb.set_title("Copy RGB")
copy_rgb.set_xlabel('Red')
copy_rgb.set_ylabel('Green')
copy_rgb.set_zlabel('Blue')
# copy_rgb.set_xlim(0, 1)
# copy_rgb.set_ylim(0, 1)
# copy_rgb.set_zlim(0, 1)


rgb.plot([0, 255], [0, 255], [0, 255], color="black", linestyle='-', linewidth=2)
copy_rgb.plot([0, 255], [0, 255], [0, 255], color="black", linestyle='-', linewidth=2)

plt.show()