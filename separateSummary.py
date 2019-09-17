import os

reference_path = './evaluation/reference'
system_path = './evaluation/system'

def write_reference_summary(file, s):
    with open(file, 'w') as f:
        f.write(s)

def write_system_summary(file, s):
    with open(file, 'w') as f:
        f.write(s)
        
with open('system_summary.txt', 'r') as sw:
    lines = sw.readlines()
    k = 0
    for line in lines:
        k += 1
        write_system_summary(os.path.join(system_path, "multidoc"+str(k)+"_system1.txt"), line)

with open('reference_summary.txt', 'r') as sw:
    lines = sw.readlines()
    k = 0
    for line in lines:
        k += 1
        write_reference_summary(os.path.join(reference_path, "multidoc"+str(k)+"_reference1.txt"), line)

