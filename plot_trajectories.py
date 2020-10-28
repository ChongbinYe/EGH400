#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import math

def plot_trajectory(x_first_part,y_first_part,x_second_part,y_second_part,xlim,ylim):
    plt.plot(x_first_part,y_first_part,color='blue')
    plt.scatter(x_first_part[-1],y_first_part[-1],s=3,color='blue')
    plt.plot(x_second_part,y_second_part,color='red')
    plt.scatter(x_second_part[-1],y_second_part[-1],s=3,color='red')
    axes = plt.gca()
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    plt.pause(0.5)
    plt.cla()
    
def plot_results_singel_lstm(x_first_part,y_first_part,x_second_part,y_second_part,x_pred,y_pred,xlim,ylim):
    plt.plot(x_first_part,y_first_part,color='blue')
    plt.scatter(x_first_part[-1],y_first_part[-1],s=3,color='blue')
    plt.plot(x_second_part,y_second_part,color='red')
    plt.scatter(x_second_part[-1],y_second_part[-1],s=3,color='red')
    plt.plot(x_pred,y_pred,color='green')
    plt.scatter(x_pred[-1],y_pred[-1],s=3,color='green')
    axes = plt.gca()
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    plt.pause(0.5)
    plt.cla()
    
    
def plot_trajectory_with_neighbours(x_first_part,y_first_part,x_second_part,y_second_part,neighbour_x,neighbour_y,weights,dummy_value=-50):
    plt.plot(x_first_part,y_first_part,color='blue')
    plt.scatter(x_first_part[-1],y_first_part[-1],s=3,color='blue')
    plt.plot(x_second_part,y_second_part,color='red')
    plt.scatter(x_second_part[-1],y_second_part[-1],s=3,color='red')
    
    # Find where the -50 (dummy point) values are and fill them with NaNs
    rows,cols=np.where(neighbour_x == dummy_value)
    neighbour_x[rows,cols]=np.nan
    neighbour_y[rows,cols]=np.nan
    
    plt.plot(neighbour_x,neighbour_y,color='purple')
    plt.scatter(neighbour_x[-1,:],neighbour_y[-1,:],s=8,color='purple')
    
    axes = plt.gca()
    axes.set_xlim([0,640])
    axes.set_ylim([0,480])
    
    for i in range(neighbour_x.shape[0]):
        for j in range(neighbour_x.shape[1]):
            #circ=plt.Circle((neighbour_x[i,j], neighbour_y[i,j]), weights[i,j]+.02, color='purple')
            circ=plt.Circle((neighbour_x[i,j], neighbour_y[i,j]),6, alpha=weights[i,j], color='purple')
            axes.add_patch(circ)
    
    
    plt.pause(1.5)
    plt.cla()
    #plt.show()
    


def newline(p1, p2):
    import matplotlib.lines as mlines
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()
    
    
    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
    
    l = mlines.Line2D([xmin,xmax], [ymin,ymax],color='black',alpha=0.15,linestyle='dashed')
    ax.add_line(l)
    
    length=20
    B_x=p2[0]
    B_y=p2[1]
    A_x=xmin
    A_y=ymin
    v_x = B_x - A_x
    v_y = B_y - A_y
    temp = v_x
    v_x = -v_y
    v_y = temp
    C_x = B_x + v_x * length
    C_y = B_y + v_y * length
    D_x = B_x + v_x * -length
    D_y = B_y + v_y * -length;
    
    l = mlines.Line2D([D_x,C_x], [D_y,C_y],color='black',alpha=0.15,linestyle='dashed')
    ax.add_line(l)
    
    return l
    
def plot_results_with_neighbours(x_first_part,y_first_part,x_second_part,y_second_part,x_pred,y_pred,neighbour_x,neighbour_y,weights,xlim,ylim,save_file_name=None,dummy_value=-50):
    plt.plot(x_first_part,y_first_part,color='blue', linewidth=2.5)
    plt.scatter(x_first_part[-1],y_first_part[-1],s=10,color='blue')
    plt.plot(x_second_part,y_second_part,color='red',linewidth=2.5)
    plt.scatter(x_second_part[-1],y_second_part[-1],s=10,color='red')
    plt.plot(x_pred,y_pred,color='green',linewidth=2.5)
    plt.scatter(x_pred[-1],y_pred[-1],s=10,color='green')
    
    # Find where the -50 (dummy point) values are and fill them with NaNs
    rows,cols=np.where(neighbour_x == dummy_value)
    neighbour_x[rows,cols]=np.nan
    neighbour_y[rows,cols]=np.nan
    
    plt.plot(neighbour_x,neighbour_y,color='purple')
    plt.scatter(neighbour_x[-1,:],neighbour_y[-1,:],s=8,color='purple')
    
    newline([x_first_part[0],y_first_part[0]],[x_first_part[-1],y_first_part[-1]])
    
    axes = plt.gca()
    
    # import matplotlib.lines as lines
    # 
    # line1 = [(x_first_part[0],y_first_part[0]), (x_first_part[-1],y_first_part[-1])]
    # (line1_xs, line1_ys) = zip(*line1)
    # axes.add_line(mlines.Line2D(line1_xs, line1_ys, linewidth=2, ))
    
    
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    
    for i in range(neighbour_x.shape[0]):
        for j in range(neighbour_x.shape[1]):
            #circ=plt.Circle((neighbour_x[i,j], neighbour_y[i,j]), weights[i,j]+.02, color='purple')
            circ=plt.Circle((neighbour_x[i,j], neighbour_y[i,j]),6, alpha=weights[i,j], color='purple')
            axes.add_patch(circ)
    
    if save_file_name is not None:
        plt.savefig(save_file_name)
    else:
        plt.pause(1.5)
    plt.cla()

if __name__ == '__main__':
    path='results/Radar/'
    fileName='model_smoothed_results.mat'
    
    data = sio.loadmat(path+fileName)
    predicted=data['predicted']
    pi_input_test= data['pi_input_test']
    
    y_test= data['y_test']
    
    x_first_part_all=pi_input_test[:,:,0]
    y_first_part_all=pi_input_test[:,:,1]
    
    x_second_part_all=y_test[:,:,0]
    y_second_part_all=y_test[:,:,1]
    
    x_pred_all=predicted[:,:,0]
    y_pred_all=predicted[:,:,1]
    
    for i in range(predicted.shape[0]):
        
        x_first_part=x_first_part_all[i,:]
        y_first_part=y_first_part_all[i,:]
        x_second_part=x_second_part_all[i,:]
        y_second_part=y_second_part_all[i,:]
        x_pred=x_pred_all[i,:]
        y_pred=y_pred_all[i,:]
        plot_results_singel_lstm(x_first_part,y_first_part,x_second_part,y_second_part,x_pred,y_pred,[-5,5],[-5, 5])
        
        

    