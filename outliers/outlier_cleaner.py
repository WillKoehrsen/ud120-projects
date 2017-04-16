#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = [(a,n, (abs(p-n)) **2) for p, a, n in zip(predictions, ages, net_worths)]

    cleaned_data = sorted(cleaned_data, key=lambda tup:tup[2])

    return cleaned_data[:80]

