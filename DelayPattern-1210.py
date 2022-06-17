#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:56:22 2021

The codes reproduce paper:
    Delay pattern estimation for signalized intersections using sampled travel times.
    by Jeff Ban, 2009
    on Transportation Research Record

@author: geneyang
"""

import pandas as pd
import numpy as np
from scipy import optimize
import math
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve
from sklearn.metrics import r2_score



attdata = pd.read_csv(r"D:\W_ProgamData\Python\dataPoisoning\delay_reproduce\arterial travel times.csv")
attdata = attdata[['Start time (sec)', 'Travel time (sec)']]
attdata.dtypes
attdata = attdata.sort_values(['Start time (sec)']) # sort by start time

## scale X
attdata['Start time (sec)'] = attdata['Start time (sec)']-attdata['Start time (sec)'].min()
attdata.reset_index(drop=True, inplace=True)
maxdiff = attdata['Travel time (sec)'].max()

print(attdata)
startTime = attdata['Start time (sec)'].to_list()
travelTime = attdata['Travel time (sec)'].to_list()

# plt.figure()
# plt.plot(startTime, travelTime,'-*')
# # attdata.plot(x='Start time (sec)', y='Travel time (sec)')


threshold_1 = 10 # to break the data and find circles
threshold_2 = 100 # not used

#%% break the data into cycles
cycles = [] #index of new cycles, each cycle is a pair of (start, end) inclusive
startindex = 0
endindex = 0
for i in range(1, len(travelTime)): # indexes of cycle borders
    if travelTime[i] - travelTime[i-1] > threshold_1:
        print(travelTime[i], " jumps from ", travelTime[i-1], ' by ', travelTime[i] - travelTime[i-1])
        endindex = i - 1
        if endindex - startindex >= 2:  # we need at least 2 points in one cycle to fit a line
            cycles.append((startindex, endindex))
        startindex = i
cycles.append((startindex, len(travelTime)-1)) # closing off the last cycle

"""
    which cycles fail to break due to SVM attack
"""
## introduce SVM's cycle breaking error: merge some circles failed to break by svm
cyclesFailed2Break = [13]
for i in cyclesFailed2Break:
    cycles[i-1] = (cycles[i-1][0], cycles[i][1])
    cycles[i] = None
cycles.remove(None)

"""
    Fitting each cycle
"""
## create a stack, so that we can do "last in and first out" operation;
# why? : we will later add new circle dynamically and process the newly added circle
cyclesASstack = [cycles[i] for i in range(len(cycles)-1, -1, -1)]
cycles = cyclesASstack

print(cycles)

#%% define function fitting two lines
def fit_2line(para):  # x0 is a_1, x1 is b_1, x2 is a_2, x3 is b_2, and x4 is m
    # objective func
    m = math.floor(para[4])
    if m < 2 or m > R - 1:
        return 1000000000000  # large enough that it will never be the min
    sum1 = ((para[0] * t[:m] + para[1] - d[:m]) ** 2).sum()
    sum2 = ((para[2] * t[m:] + para[3] - d[m:]) ** 2).sum()
    return sum1 + sum2


def Fcontr(para):  # x0 is a_1, x1 is b_1, x2 is a_2, x3 is b_2, and x4 is m
    # constraint
    m = math.floor(para[4])
    boundINgroup = (1 - theta) * t[m - 1] + theta * t[m]
    return para[0] * boundINgroup + para[1] - (para[2] * boundINgroup + para[3])


constr = {'type': 'eq', 'fun': Fcontr}


#%% main function for fittign all circle and visulization

# cycles = [cycles[0]] # test: pull out one circle for test; i am testing on the 3rd circle

plt.figure()
plt.scatter(startTime, travelTime, marker='x', s=15, color='darkblue', alpha=0.7)

theta = 0.5
# cycle = (82, 84) # (28,35) # test
pre_cycle = None

# cycle = cycles[13] # for test
rr = 0
while len(cycles):
    cycle = cycles.pop()
    # x and y in one cycle
    t, d = np.array(startTime[cycle[0]: cycle[1]+1]), np.array(travelTime[cycle[0]: cycle[1]+1])
    R = len(t)
    print(t), print(d)

    leftBoundOUTgroup = startTime[0] if cycle[0] - 1 < 0 else int((1 - theta) * startTime[cycle[0] - 1] + theta * startTime[cycle[0]])
    rightBoundOUTgroup = startTime[-1] if cycle[1] + 1 >= len(startTime) else int((1 - theta) * startTime[cycle[1]] + theta * startTime[cycle[1]+1])


    if R > 2:
        ## fit with two lines
        # Initial values. It is reasonable to set -x+40000 for the linear equations,
        # and assume that the two lines are cut somewhere between the first and last point: m
        a1, b1, a2, b2, m = -1, 40000, -1, 40000, 1 + int(R/2)
        para = np.array([a1, b1, a2, b2, m])
        result2line = optimize.minimize(fit_2line, para, constraints=constr)
        a1, b1, a2, b2, m = result2line.x
        m = int(m)

        error2lines = result2line.fun
        predicted2line = list(a1 * t[:m] + b1)
        predicted2line.extend(list(a2 * t[m:] + b2))
        R2for2lines = r2_score(d, predicted2line)
    else:
        R2for2lines = -1

    ## fit t, d with one line
    result1line = LinearRegression().fit(t.reshape(-1, 1), np.array(d))
    error1line = ((d - result1line.predict(t.reshape(-1, 1)))**2).sum()
    R2for1line = result1line.score(t.reshape(-1, 1), np.array(d))#R^2 score: r2_score(d, result1line.predict(t.reshape(-1, 1)))
    a, b = result1line.coef_[0], result1line.intercept_

    # if error2lines > error1line: we use/plot the results from one-line-fitting; else: use/plot the two-line-fitting
    if R<=2 or (R2for2lines - R2for1line<=0.1 and (t[-1]-t[0]) <= threshold_2): # instead of comparing func errors, we compare R^2, an indicator of good fitness of a model
        print("For cycle " + str(cycle) + " of length " + str(R) + ", we have 1 line: ")
        print("a: ", a, " b: ", b)
        if a > 0: a, b = 0, 20
        # plot red duration
        plt.hlines(y=20, xmin=leftBoundOUTgroup+rr, xmax=leftBoundOUTgroup+leftBoundOUTgroup * a + b-20, linewidth=3, colors='red', alpha=0.7)
        # plot the slop
        root_intersect_y20 = int(fsolve(lambda ti: a * ti + b - 20, rightBoundOUTgroup)) # min: the root or the bound
        if root_intersect_y20 < rightBoundOUTgroup:
            rightBoundOUTgroup = root_intersect_y20 # rightBoundOUTgroup = min(root_intersect_y20, rightBoundOUTgroup)
            rr = 0
        else:
            rr = rightBoundOUTgroup * a + b-20
        plt.plot(np.array(range(leftBoundOUTgroup, rightBoundOUTgroup)),
                 a * np.array(range(leftBoundOUTgroup, rightBoundOUTgroup)) + b, 'b-', linewidth=1, alpha=0.7)
        plt.vlines(leftBoundOUTgroup, ymin=20, ymax=leftBoundOUTgroup * a + b, colors='darkblue', linewidth=1, alpha=0.7)
    else:
        print("For cycle ", cycle, " of length ", R, "we have 2 lines: ")
        print("a1: ", a1, " b1: ", b1, " a2: ", a2, " b2: ", b2, " m: ", m)
        if a1 > 0: a1, b1 = 0, 20
        if a2 > 0: a2, b2 = 0, 20
        boundINgroup = int((1 - theta) * t[m - 1] + theta * t[m]) # to plot the break of the two lines in the group
        # if a circle is too long, we break it again; the two cycles are added to the cycle_stack
        if (boundINgroup - t[0] > threshold_2) or (t[-1] - boundINgroup) > threshold_2:
            cycles.append((cycle[0]+m, cycle[1]))
            cycles.append((cycle[0], cycle[0]+m-1))
            continue
        # plot red duration
        plt.hlines(y=20, xmin=leftBoundOUTgroup+rr, xmax=leftBoundOUTgroup+leftBoundOUTgroup * a1 + b1-20, linewidth=3, colors='red', alpha=0.7)
        boundINgroup = min(int(fsolve(lambda ti: a1 * ti + b1 - 20, rightBoundOUTgroup)), boundINgroup) # stop at where it intersects y=20
        root_intersect_y20 = int(fsolve(lambda ti: a2 * ti + b2 - 20, rightBoundOUTgroup))
        if root_intersect_y20 < rightBoundOUTgroup:
            rightBoundOUTgroup = root_intersect_y20 # rightBoundOUTgroup = min(root_intersect_y20, rightBoundOUTgroup) # stop at where it intersects y=20
            rr = 0
        else:
            rr = rightBoundOUTgroup * a1 + b1-20

        t1, t2 = np.array(range(leftBoundOUTgroup, boundINgroup)), np.array(range(boundINgroup, rightBoundOUTgroup))
        print("lengths of two circles: ", (len(t1), len(t2)))

        plt.plot(t1, a1 * t1 + b1, 'b-', linewidth=1, alpha=0.7)
        plt.plot(t2, a2 * t2 + b2, 'b-', linewidth=1, alpha=0.7)

        ## vertical line on the left breaking groups
        plt.vlines(leftBoundOUTgroup, ymin= 20, ymax=leftBoundOUTgroup*a1+b1, linewidth=1, colors='darkblue', alpha=0.7)
        ## vertical line in the middle breaking the two fitted lines
        plt.vlines(boundINgroup, ymin=20, ymax=boundINgroup * a1 + b1, linewidth=1, colors='darkblue', alpha=0.7)

    # pre_cycle = cycle

plt.ylim([0, maxdiff+5])
plt.hlines(y=20, xmin=startTime[0], xmax=startTime[-1], linewidth=1, colors='darkblue', alpha=0.7)
plt.ylabel('Travel time (sec)')
plt.xlabel('Start time (sec)')
plt.savefig('delay_estimate_all')
plt.show()


"""
    According to paper: 
        Real time queue length estimation for signalized intersections using travel times from mobile sensors 
        by Jeff Ban 2011, on Transportation Research Part C: Emerging Technologies
    once we fit the delay pattern, the queue length is computed using the root of the fitted line, which is
    propotional to: 
        - intersect / slope 
    This allows us to compute percent change of queue length.
    More information (e.g., traffic flow rate, wave speed) is needed to compute the absolute value of queue length. 
"""

    
