from __future__ import print_function, division
import os
import sys
import json
import pandas as pd

class_dict = {
1: "BrushHair",
2: "Catch",
3: "Clap",
4: "ClimbStairs",
5: "Golf",
6: "Jump",
7: "KickBall",
8: "Pick",
9: "Pour",
10: "PullUp",
11: "Push",
12: "Run",
13: "ShootBall",
14: "ShootBow",
15: "ShootGun",
16: "Sit",
17: "Stand",
18: "SwingBaseball",
19: "SyntheticBumpIntoEachOther",
20: "SyntheticCarHit",
21: "SyntheticCrawl",
22: "SyntheticDiveFloor",
23: "SyntheticFlee",
24: "SyntheticHop",
25: "SyntheticLegSplit",
26: "SyntheticLimp",
27: "SyntheticMoonwalk",
28: "SyntheticStagger",
29: "SyntheticSurrender",
30: "SyntheticWalkHoldHands",
31: "SyntheticWalkingHug",
32: "SyntheticWalkTheLine",
33: "Throw",
34: "Walk",
35: "Wave",
}


# models = [Model_Girl, Model_Boy, Model_F_Grandmother, MaleCharacter, MixamoVincent5kPoly, Vincent,
# Civilian_Female02, MixamoCarlHiRes, Alexis, Joan, Justin, Civilian_Male04, NewsReader, Civilian_Male01,
# Civilian_Female01, Prisoner_01, Prisoner_02, Civilian_Janitor, Security_officer_Male01, Civilian_Father]

# train_models = [Model_F_Grandmother, MaleCharacter, MixamoVincent5kPoly, Vincent,
# Civilian_Female02, MixamoCarlHiRes, Joan, Civilian_Male04, NewsReader, Civilian_Male01,
# Civilian_Female01, Prisoner_01, Prisoner_02, Civilian_Janitor, Security_officer_Male01, Civilian_Father]

# val_models = [Alexis, Justin]

# test_models = [Model_Girl, Model_Boy]

#Catch/b3050-House-Dawn-Foggy-33_02-Model_Girl-Indoor-o-7e05119cbac5858f4c13461f9ca93b7c
# naming configuration:
# action_dir/key-Environment-Day_Phase-Weather-#_#-Human_Model-Camera-Motion_Variation-ID#


def remove_data_bias(data_list, str_remove):
    delete_list = []
    for i, data in enumerate(data_list):
        if data.count(str_remove) > 0:
            delete_list.append(i)
    for j in reversed(range(len(delete_list))):
        del data_list[delete_list[j]]
    return data_list

def create_train_list(train_list, train_removals):
    for str_remove in train_removals:
        train_list = remove_data_bias(train_list, str_remove)

    with open(csv_dir_path+'/train_list.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(train_list))
        myfile.write('\n')


def create_val_list(val_list, val_removals):
    for str_remove in val_removals:
        val_list = remove_data_bias(val_list, str_remove)

    with open(csv_dir_path+'/val_list.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(val_list))
        myfile.write('\n')

def create_test_list(test_list, test_removals):
    for str_remove in test_removals:
        test_list = remove_data_bias(test_list, str_remove)

    with open(csv_dir_path+'/test_list.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(test_list))
        myfile.write('\n')

def append_category(data_list):
    for i, data in enumerate(data_list):
        for action_catagory in class_dict:
            if data[:len(class_dict[action_catagory])] == class_dict[action_catagory]:
                data_list[i] += ' ' + str(action_catagory)
    return data_list

def create_total_list(csv_dir_path, data_dir_path, list_of_removals):
    data_list = []
    count = 0
    # Create list of video directory names
    for root, dirs, _ in os.walk(data_dir_path):
        if len(root[len(data_dir_path)+1:]) > 65:
            count = count +1
            data_list.append(root[len(data_dir_path)+1:])

    # Append action catagory to list of each directory
    data_list = append_category(data_list)

    # remove unwanted example attributes
    for str_remove in list_of_removals[3]:
        data_list = remove_data_bias(data_list, str_remove)

    # write .txt file to csv_dir_path
    with open(csv_dir_path+'/data_list_full.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(data_list))
        myfile.write('\n')


    # separate train / val / test sets
    create_train_list(data_list.copy(), list_of_removals[0])
    create_val_list(data_list.copy(), list_of_removals[1])
    create_test_list(data_list.copy(), list_of_removals[2])


if __name__ == '__main__':
    csv_dir_path = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos/phavTrainTestlist'
    data_dir_path = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos'

    #all_removals = ['Dusk', 'Dawn', 'Overcast', 'Rainy', 'Foggy', 'Cloudy']  #onlydayclear
    #all_removals = ['Overcast', 'Rainy', 'Foggy', 'Cloudy']    #onlyclear
    all_removals = ['Rainy', 'Foggy']

    train_removals = ['Alexis', 'Justin', 'Model_Girl', 'Model_Boy']

    val_removals = ['Model_Girl', 'Model_Boy', 'Model_F_Grandmother', 'MaleCharacter', 'MixamoVincent5kPoly', 'Vincent',
                     'Civilian_Female02', 'MixamoCarlHiRes', 'Joan','Civilian_Male04', 'NewsReader',
                     'Civilian_Male01', 'Civilian_Female01', 'Prisoner_01', 'Prisoner_02', 'Civilian_Janitor',
                     'Security_officer_Male01', 'Civilian_Father']

    test_removals = ['Model_F_Grandmother', 'MaleCharacter', 'MixamoVincent5kPoly', 'Vincent',
                     'Civilian_Female02', 'MixamoCarlHiRes', 'Alexis', 'Joan', 'Justin', 'Civilian_Male04', 'NewsReader',
                     'Civilian_Male01', 'Civilian_Female01', 'Prisoner_01', 'Prisoner_02', 'Civilian_Janitor',
                     'Security_officer_Male01', 'Civilian_Father']

    list_of_removals = [train_removals, val_removals, test_removals, all_removals]
    data_list = create_total_list(csv_dir_path, data_dir_path, list_of_removals)
