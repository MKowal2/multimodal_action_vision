# averages all test set accuracies in the listed directory
import os
import csv
rootdir = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/fusion_outputs/kinetics_weight2_val_fusion_x5_nov12'

# the last 5 characters are unique -> ...phav_i3d_RC
# R, D, C, I, F
# RF, RD, RC, RI, FD, FC, IF, DC, DI, CI
strings = [#'R', 'F', 'D', 'I', 'C',
           'RF', 'RD', 'RI', 'RC',  'DF', 'IF', 'CF', 'DI', 'DC', 'CI',
           'RDF', 'RIF', 'RCF', 'RDI', 'RDC', 'RCI',
           'DIF', 'DCF', 'CIF', 'DCI',
           'RDIF', 'RDCF', 'RCIF', 'RDCI', 'DCIF',
           'RDCIF'
           ]

num_outputs = 5
dict1 = {}
for x in strings:
    dict1[x] = 0

dict2 = {}
dict3 = {}
dict4 = {}
for x in strings:
    dict2[x] = 0
    dict3[x] = []
    dict4[x] = {}
    for i in range(100):
        dict4[x][i] = 0

for string in strings:
    dir_name_end = 'kinetics_i3d_'+ string
    for subdir, dirs, files in os.walk(rootdir):
        if subdir[-5:] == dir_name_end[-5:]:
            print(subdir)
            for file in files:
                if file == 'info.txt':
                    with open(subdir + '/info.txt', "r") as f:
                        results = f.read().split(',')
                        acc = float(results[4])
                        loss = float(results[3])
                        dict1[string] = dict1[string] + float(acc)
                        dict2[string] = dict2[string] + float(loss)
                        dict3[string].append(acc)
                if file == 'class_acc.txt':
                    with open(subdir + '/class_acc.txt', "r") as f:
                        for lines in f.readlines():
                            idx = float(lines.split(',')[0])
                            acc = float(lines.split(',')[1])

                            dict4[string][idx] += acc



for key in dict1:
    dict1[key] = (dict1[key] / num_outputs)*100
    dict2[key] = dict2[key] / num_outputs
    for i in range(100):
        dict4[key][i] = dict4[key][i] / num_outputs

# for key in dict3:
#     sum = 0
#     for i in range(num_outputs):
#         # sum of each element subtracted from mean
#         sum += (dict3[key][i] - dict1[key])**2
#     dict3[key] = sum / num_outputs



with open(rootdir + '/ave_acc1.txt', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=strings)
    writer = csv.writer(csvfile)
    for key, value in dict1.items():
        writer.writerow([key, value])

with open(rootdir + '/ave_loss1.txt', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=strings)
    writer = csv.writer(csvfile)
    for key, value in dict2.items():
        writer.writerow([key, value])

# with open(rootdir + '/std1.txt', 'w') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=strings)
#     writer = csv.writer(csvfile)
#     for key, value in dict3.items():
#         writer.writerow([key, value])

with open(rootdir + '/per_class_acc.txt', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=strings)
    writer = csv.writer(csvfile)
    for key, value in dict4.items():
        writer.writerow([key, value])


