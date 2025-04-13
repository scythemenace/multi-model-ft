import csv
import random

# Resource used: https://docs.python.org/3/library/csv.html
testfile = 'testing.csv'  
baselineFile = 'simple-baseline_results.csv'
outputRow= []

with open(testfile, mode='r') as f:
    reader  = csv.reader(f)
    for row in reader :
        # Random Baseline
        outputRow.append([random.choice([0, 1])])

# Output file
with open(baselineFile, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(outputRow)

