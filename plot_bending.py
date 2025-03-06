import matplotlib.pyplot as plt
import numpy as np
import os
def plot_data_from_txt(file_path,title_y_axis,title_x_axis,temperature,color,sort):
    x_axis = []
    y_axis = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            x_axis.append(float(x))
            y_axis.append(float(y))
    sorted_data = sorted(zip(x_axis, y_axis))  # Sort the list of tuples based on the x-values
    x_axis_sorted, y_axis_sorted = zip(*sorted_data)
    if sort:
        x_axis,y_axis = x_axis_sorted,y_axis_sorted
    plt.plot(x_axis, y_axis, marker='o', linestyle='-', color=color)
    plt.title(title_y_axis+' vs. '+title_x_axis, fontsize=16)
    plt.xlabel(title_x_axis, fontsize=14)
    plt.ylabel(title_y_axis, fontsize=14)



plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plot_data_from_txt("bending_angle_data.txt","Temperature","Bending Angle","Angle Vs. Temperature","r",False)
plt.figure(figsize=(10, 6))
plot_data_from_txt("deformation_data1.txt","x axis","y axis","Nodes final position","r",True)
plot_data_from_txt("deformation_data2.txt","x axis","y axis","Nodes final position","b",True)

plt.show()