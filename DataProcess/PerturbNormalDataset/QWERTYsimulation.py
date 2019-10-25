# simulate pressing data -> letter probability from the paper
# Typing on an Invisible Keyboard, CHI 2018
# author : Mingrui Ray Zhang

from numpy.random import multivariate_normal
import numpy as np

# x.neg -> left; x.pos -> right
# y.neg -> above; y.pos -> velow
mu_x = {
    'a': -24.37, 'b': -12.4,  'c': -6.43, 'd': -6.15,  'e': -15.23, 'f': -10.28, 'g': -12.77,
    'h': -10.62, 'i': -10.71, 'j': -5.58, 'k': -12.95, 'l': -14.01, 'm': -13.62, 'n': -13.86,
    'o': -14.69, 'p': -14.88, 'q':  4.07, 'r':  -9.43, 's': -14.5,  't': -6.14, 
    'u':  -9.24, 'v': -10.5,  'w': -6.69, 'x':   2.38, 'y': -8.77,  'z': -15.44
}

mu_y = {
    'a': 21.33, 'b': 19.89, 'c': 19.46, 'd': 18.78,  'e': 22.87, 'f': 17.65, 'g': 23.93,
    'h': 22.34, 'i': 22.08, 'j': 16.72, 'k': 23.40,  'l': 23.95, 'm': 16.36, 'n': 23.84,
    'o': 29.43, 'p': 31.42, 'q': 20.80, 'r': 24.31,  's': 21.25, 't': 24.06, 
    'u': 24.18, 'v': 14.28, 'w': 27.33, 'x': -10.26, 'y': 27.67, 'z': 14.97
}

sigma_x = {
    'a': 33.49, 'b': 29.43, 'c': 27.40, 'd': 29.06, 'e': 29.44, 'f': 22.60, 'g': 27.07,
    'h': 29.56, 'i': 29.56, 'j': 25.17, 'k': 29.13, 'l': 27.49, 'm': 29.96, 'n': 27.32,
    'o': 26.65, 'p': 29.16, 'q': 48.88, 'r': 31.60, 's': 30.90, 't': 26.93, 
    'u': 26.96, 'v': 24.90, 'w': 22.70, 'x': 17.96, 'y': 30.38, 'z': 28.87
}

sigma_y = {
    'a': 27.21, 'b': 29.86, 'c': 28.17, 'd': 27.71, 'e': 22.20, 'f': 21.77, 'g': 26.24,
    'h': 29.41, 'i': 23.84, 'j': 20.44, 'k': 23.55, 'l': 24.33, 'm': 26.62, 'n': 24.93,
    'o': 24.52, 'p': 23.23, 'q': 20.41, 'r': 21.60, 's': 24.03, 't': 22.60, 
    'u': 23.36, 'v': 24.16, 'w': 20.57, 'x': 19.59, 'y': 26.89, 'z': 21.90
}

rho = {
    'a': 0.16, 'b': 0.40, 'c': 0.11, 'd': 0.05, 'e': 0.04, 'f': 0.01, 'g': -0.36,
    'h': 0.11, 'i': 0.30, 'j': 0.15, 'k': 0.30, 'l': 0.22, 'm': 0.24, 'n': 0.13,
    'o': 0.26, 'p': 0.38, 'q': 0.14, 'r':-0.12, 's': 0.22, 't':-0.09, 
    'u': 0.03, 'v':-0.03, 'w':-0.03, 'x':-0.19, 'y':-0.03, 'z': 0.17
}

# key size / 2
W, H = 54.0, 67.5 
TOTPTS = 10000000

keys = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', \
          'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', \
               'z', 'x', 'c', 'v', 'b', 'n', 'm']

errors = {}

for i in range(len(keys)):
    key = keys[i]
    mean = [mu_x[key], mu_y[key]]
    cov  = [[sigma_x[key]**2, rho[key]*sigma_x[key]*sigma_y[key]], \
            [rho[key]*sigma_x[key]*sigma_y[key], sigma_y[key]**2]]

    pts = np.array(multivariate_normal(mean, cov, TOTPTS))
    left_over  = pts[pts[:,0] < -W]
    right_over = pts[pts[:,0] > W]
    above_over = pts[pts[:,1] < -H]
    below_over = pts[pts[:,1] > H]

    '''
    touch area related to a key (middlecorrect is the key space)
        mid_above_left     |    mid_above_right
        left_above | middle_above | right above
        ----------------------------------------
        left_middle| middlecorrect| right_middle
        ----------------------------------------
        left_below | middle_below | right_below
        mid_below_left     |    mid_below_right
    '''

    left_above = len(left_over[left_over[:,1] < -H])
    left_below = len(left_over[left_over[:,1] > H])

    right_above = len(right_over[right_over[:,1] < -H])
    right_below = len(right_over[right_over[:,1] > H])

    middle_above = len(above_over) - left_above - right_above
    middle_below = len(below_over) - left_below - right_below

    left_middle = len(left_over) - left_above - left_below
    right_middle = len(right_over) - right_above - right_below

    mid_above_left = len(above_over[above_over[:,0] < 0])
    mid_above_right = len(above_over) - mid_above_left

    mid_below_left = len(below_over[below_over[:,0] < 0])
    mid_below_right = len(below_over) - mid_below_left

    errors[key] = {}
    # has right key
    if not (i == 9 or i == 18 or i == 25):
        errors[key][keys[i+1]] = right_middle / TOTPTS

    # has left key
    if not (i == 0 or i == 10 or i == 19):
        errors[key][keys[i-1]] = left_middle / TOTPTS

    # first row
    if i <= 9:
        # has right below key
        if i < 9:
            errors[key][keys[i+10]] = mid_below_right / TOTPTS
        # has left below key
        if i > 0:
            errors[key][keys[i+9]] = mid_below_left / TOTPTS
    
    # second row
    elif i <= 18:
        # left above
        errors[key][keys[i-10]] = left_above / TOTPTS
        # right above
        errors[key][keys[i-9]] = right_above / TOTPTS
        # has below key
        if i > 10 and i < 18:
            errors[key][keys[i+8]] = middle_below / TOTPTS
        # has left below key
        if i > 11:
            errors[key][keys[i+7]] = left_below / TOTPTS
        # has right below key
        if i < 17:
            errors[key][keys[i+9]] = right_below / TOTPTS

    # third row
    else:
        # left above
            errors[key][keys[i-9]] = left_above / TOTPTS
        # right above 
            errors[key][keys[i-7]] = right_above / TOTPTS
        # above
            errors[key][keys[i-8]] = middle_above / TOTPTS

for key in errors:
    for to_key in errors[key]:
        print ('%s --> %s %.6f' % (key, to_key, errors[key][to_key]))
