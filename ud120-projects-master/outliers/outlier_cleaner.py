#!/usr/bin/python

from numpy import array
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    temp = [(a[0], n[0], n[0] - p[0]) for a, n, p in zip(ages, net_worths, predictions)]
    t = sorted(temp, key = lambda s: abs(s[2]), reverse=True)
    a = int(len(t)*0.1)
    cleaned_data = t[a:]

    ### your code goes here

    
    return cleaned_data

