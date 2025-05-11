import torch
import numpy as np
import os, sys

#from torchvision import datasets, models
#import torchvision.transforms as transforms
#from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.data import Dataset
#import natsort
#from PIL import Image
import pandas as pd
from openpyxl import load_workbook
#from asammdf import MDF # read sql MDF dataset files
import pyodbc # connect to SQL SERVER
import gc
import time
import datetime
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import re


#https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
      R = 6372800 # 3959.87433 is in miles.  For Earth radius in kilometers use 6372.8 km
      dLat = radians(lat2 - lat1)
      dLon = radians(lon2 - lon1)
      lat1 = radians(lat1)
      lat2 = radians(lat2)
      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2*asin(sqrt(a))
      return R * c

def distance(s_lat, s_lng, e_lat, e_lng):
   # approximate radius of earth in meters
   R = 6373000.0 # 6373.0 in km
   s_lat = s_lat*np.pi/180.0
   s_lng = np.deg2rad(s_lng)
   e_lat = np.deg2rad(e_lat)
   e_lng = np.deg2rad(e_lng)
   d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
   return 2 * R * np.arcsin(np.sqrt(d))

def isint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# node_data has header, but link data doesn't have.
# nodes in noda_data are the only one relevant, all else are dissmised, e.g. Ayalon BTAY nodes.
# if you don't want a header row or index column, use to_csv("...", header=None, index=None)


def Build_Adjacency_Matrix(inp_file, nodes_file, out_adj_nodes, out_link_file):
    # Create Adj. matrix file of nodes and links table (name,length) files, from the raw data file, rows=from, columns=to.
    FileReader = pd.read_csv(inp_file, chunksize=100000, header=None)  # the number of rows per chunk #, index_col=0)
    node_data = pd.read_csv(nodes_file, delimiter=',')  # , error_bad_lines=False)
    node_name = node_data[node_data.columns[1]].values
    node_x = node_data[node_data.columns[13]].values
    node_y = node_data[node_data.columns[14]].values
    nodes_length = len(node_name)  # B column has list of nodes.
    node_table = np.zeros((nodes_length, nodes_length), dtype=np.float32)    # table w/o lengths (0,1)
    node_table_l = np.zeros((nodes_length, nodes_length), dtype=np.float32)  # table with lengths
    num_chunks = sum(1 for row in open(inp_file, 'r')) // 100000
    chunk = -1
    add_col = 0
    for df in FileReader:
        chunk += 1
        print(" chunk = "+str(chunk)+"/"+str(num_chunks))        # df is a chunk of data
        if (chunk == 0) and (df[0].values[0] == df[2].values[0]):
            add_col = 1  # add column to the next ones
        enter_node = df[4 + add_col].values
        exit_node = df[5 + add_col].values
        tripStatus = df[15 + 4 + add_col].values
        curr_index = -1
        for x in enter_node:
            curr_index = curr_index + 1  # index of the current row in original data
            if tripStatus[curr_index] == "Valid" and not(np.isnan(enter_node[curr_index])) and not(np.isnan(exit_node[curr_index])):
                node_begin = np.where(node_name == enter_node[curr_index])
                node_end = np.where(node_name == exit_node[curr_index])
                node_begin = np.squeeze(node_begin)
                node_end = np.squeeze(node_end)
                if (node_table[node_begin, node_end]==0) and (node_begin.size>0) and (node_end.size>0):
                    #if (enter_node[curr_index]=="TA193") and (exit_node[curr_index]=="TA5"):
                    #     print(curr_index)
                    node_table[node_begin, node_end] = 1  # from node to node
                    node_table_l[node_begin, node_end] = haversine(node_y[node_begin],node_x[node_begin],node_y[node_end],node_x[node_end])
                    node_table_l[node_end, node_begin] = node_table_l[node_begin, node_end]  # length is symmetric.
                    #node_table_l[node_begin, node_end] = np.sqrt((node_x[node_end] - node_x[node_begin]) ** 2 + (node_y[node_end] - node_y[node_begin]) ** 2)  # distance from node to node

    links = []
    links_length = []
    # loop over the whole matrix, so that bi-derational links will be counted seperatly, i.e. twice if necessary.
    # Why? cause the links in one direction and the opposite if the same link are different and independent (directly at least).
    for row in range(nodes_length):
        for column in range(nodes_length):
            if node_table[row,column]>0:
                links.append(node_name[row]+node_name[column])
                links_length.append(node_table_l[row,column])

    pd.DataFrame(node_table).to_csv(out_adj_nodes, header=None, index=None)  # export the node_table 2D array.
    writer = pd.ExcelWriter(out_link_file, engine='openpyxl')
    pd.DataFrame(np.transpose([links,links_length])).to_excel(writer, header=None, index=None)  # export links meta_data
    writer.save()
    writer.close()


def Build_Adjacency_Matrix2(inp_file, links_file, out_file):
    # Create Adj. matrix of links file, from the raw data file, rows=from link, columns=to link.
    FileReader = pd.read_csv(inp_file, chunksize=100000, header=None)  # the number of rows per chunk #, index_col=0)
    links = pd.read_excel(links_file, header=None)  # , error_bad_lines=False)
    links = links[0].values
    link_adj_mat = np.zeros((len(links), len(links)), dtype=np.int32)
    num_chunks = sum(1 for row in open(inp_file, 'r')) // 100000
    chunk = -1
    add_col = 0
    for df in FileReader:
        chunk += 1
        print(" chunk = "+str(chunk)+"/"+str(num_chunks))        # df is a chunk of data
        if (chunk == 0) and (df[0].values[0] == df[2].values[0]):
            add_col = 1  # add column to the next ones
        before_node = df[3 + add_col].values
        enter_node = df[4 + add_col].values
        exit_node = df[5 + add_col].values
        after_node = df[6 + add_col].values
        tripStatus = df[15 + 4 + add_col].values
        curr_index = -1
        for x in enter_node:
            curr_index = curr_index + 1  # index of the current row in original data
            if tripStatus[curr_index] == "Valid" and not(np.isnan(enter_node[curr_index])) and not(np.isnan(exit_node[curr_index])):
                link2 = np.squeeze(np.where(links == enter_node[curr_index] + exit_node[curr_index]))
                if (not(np.isnan(before_node[curr_index]))) and (link2.size>0):
                    link1 = np.squeeze(np.where(links == before_node[curr_index] + enter_node[curr_index]))
                    if (link1.size>0):
                         link_adj_mat[link1, link2] = 1
                if (not(np.isnan(after_node[curr_index]))) and (link2.size>0):
                    link3 = np.squeeze(np.where(links == exit_node[curr_index] + after_node[curr_index]))
                    if (link3.size>0):
                         link_adj_mat[link2, link3] = 1

    pd.DataFrame(link_adj_mat).to_csv(out_file, header=None, index=None)  # export meta_data


def Assign_DF(dff, indx, vallue):
    xx = [indx] + vallue
    if indx in dff[:,0]:
        inx = np.squeeze(np.where(dff[:, 0] == indx))
        dff[inx,:] = xx
    else:
        xx = np.array(xx).reshape((1, dff.shape[1]))
        dff = np.concatenate((dff, xx), axis=0)
    return dff


def CompareArrays(A,B):
    return np.array_equal(A,B) and np.array_equiv(A,B) and (A==B).all()

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()




## Post-processing Functions:

def Find_Ranges(df, symb, max_NaNs):
    # search for continuous range of not nan/symb rows
    rl = range(len(df))
    ranges_start = [[] for _ in rl]
    ranges_end = [[] for _ in rl]
    for indx in rl:
        # Calculate streaks, filter out streaks that are too short, apply global mask
        if symb is np.nan:# or symb==float('nan'):
            nan_spots = np.where(np.isnan(df[indx]))  # get indexes of all NaNs in the array
        else:
            nan_spots = np.where(df[indx]==symb)  # get indexes of all symb in the array
        if nan_spots[0].size==0:
            ranges_start[indx] = []
            ranges_end[indx] = []
            continue
        diff = np.diff(nan_spots)[0]  # get distance between NaNs indexes (1=succesive NaNs)
        streaks = np.split(nan_spots[0], np.where(diff != 1)[0] + 1)  # split all sequences of NaNs
        long_streaks = [streak for streak in streaks if len(streak) < max_NaNs]
        if len(long_streaks)==0:
            ranges_start[indx] = []
            ranges_end[indx] = []
            continue
        ff = list(np.hstack([[streak[0], streak[len(streak) - 1] + 1] for streak in long_streaks]))
        #ff = [0] + ff + [len(df[indx])]
        ranges_start[indx] = ff[::2]
        ranges_end[indx] = ff[1::2]
    return ranges_start,ranges_end

def Post_process_links(out_file, nodes_file, links_of_interest):
    # Objective: reduce the full data to partial one, by eliminiating all links but the ones specified in "links_of_interest"
    # These are the indexes of the rows of the links you're interested in, in the links file, e.g. "adj04_links.xlsx"
    # Choose the indexes from there.
    # This function updates the out_file (which is the original post-proccesed file) into out_file + "_linkset" file,
    # and updates its adjacency matrix and its links array.

    node_data = np.load(nodes_file, allow_pickle=True).item()
    node_data = node_data["Links_by_Name"]
    nodes_length = node_data.shape[0]
    indxs = np.where(node_data == links_of_interest)
    # out_file:
    df = np.load(out_file, allow_pickle=True).item()
    # Go over all sheets in the file:
    for curr_sheet in list(df.keys()):
        # don't need to change "Info" sheet or "Info_Links"
        if (curr_sheet[0:4]=="Info"):
            continue
        df[curr_sheet] = df[curr_sheet][:,indxs]  # here the links are the columns in the file
    new_adj = df["AdjMatrix"][indxs, :]  # here the links are the rows in the file
    new_adj = new_adj[:,indxs]  # and the columns in the file
    df["Info_Links"] = df["Info_Links"][indxs, :]
    np.save(os.path.splitext(out_file)[0] + "_linkset" + os.path.splitext(out_file)[1], df)

    print("Finished Post_process different links")


def interpolate_array(df_curr_org):
    #from scipy.interpolate import interp1d

    # add 1st row to be of zeros, for smooth interpolation, and removing nan's in the 1st row:
    #aa = np.zeros((1, df_curr.shape[1]))
    df_curr = df_curr_org.copy()
    # if np.isnan(df_curr_org[0]):
    #     df_curr = np.concatenate(([0], df_curr), axis=0)  # [aa, df_curr]
    # if np.isnan(df_curr_org[-1]):
    #     df_curr = np.concatenate((df_curr, [0]), axis=0)  # [aa, df_curr]
    nans, x = np.isnan(df_curr), lambda z: z.nonzero()[0]
    df_curr[nans] = np.interp(x(nans), x(~nans), df_curr[~nans])
    # df_curr = df_curr.interpolate(method='linear', axis=0)  # only interpolate for each node speratly, not accross nodes.
    # xvals = np.linspace(0, len(df_curr)-1, len(df_curr))
    # fintr = interp1d(xvals, df_curr, axis=0)  # np.interp(xvals, xvals, df_curr)
    # df_curr = fintr(xvals)
    # df_curr = df_curr[1:]
    # if np.isnan(df_curr_org[0]):
    #     df_curr = df_curr[1:]
    # if np.isnan(df_curr_org[-1]):
    #     df_curr = df_curr[:-1]
    return df_curr

def Post_process_smooth():   #  out_file , sheet_name, is_interpolate, is_average):
    # Process smoothing on one sheet_name (to interpolate and/or to average)
    # copy all in_file sheets to out_file.
    # No data:
    # "TravelTime": Inf
    # "Velocity": Inf
    # "Density" : 0
    # "Plans": 0
    # "Flow": 0

    out_file = data_fname_output

    in_file = os.path.splitext(out_file)[0] + "_04" + os.path.splitext(out_file)[1]  # Always create new file
    # if not os.path.exists(in_file):  # if 05 don't exists - it's ok, it's not mandatory (it's optional)
    #     in_file = os.path.splitext(out_file)[0] + "_04" + os.path.splitext(out_file)[1]
    out_file = os.path.splitext(out_file)[0] + "_05" + os.path.splitext(out_file)[1]
    df = np.load(in_file, allow_pickle=True).item()
    '''
    # nodes_of_interest = [77,        78,             79,               80,              81,              83,         85,          183,          185]
    # as in excel file (so need to decrease by 1):
    links_of_interest = [88, 241, 248, 49, 173, 245, 251, 16, 34, 249, 256, 27, 116, 216, 252, 28, 117, 176, 262, 6,
                        260, 266, 12, 187, 263, 153, 254, 258, 24, 239]
    # links_of_interest = [2, 5, 24, 51, 66, 92, 104, 107, 152, 156, 159, 160, 161, 162, 164, 166, 167, 168]
    links_of_interest = [x - 2 for x in links_of_interest]
    '''
    #max_diff = 20     # maximum remainder after filling the gap with same-size cycles
    # First combine input and output sheets
    smooth_sheets = input_sheets.copy()
    for sheet_name in output_sheets:
        if sheet_name not in smooth_sheets:
            smooth_sheets.append(sheet_name)

    indx = 0
    for sheet_name in smooth_sheets:
        if (sheet_name in BT_sheets) or (sheet_name in SP_sheets):
            is_interpolate, is_average = smoothings[indx]  # smoothings arranged in a input-then-output order only for the valid features

            df_sheet = df[sheet_name]  # [df[sheet_name][i] for i in links_of_interest]
            if sheet_name == "Flow":
                df_sheet = [np.array(x, dtype=float) for x in df_sheet]

            if (is_interpolate):  # and (sheet_name in continuous_sheets):  CAN BE ANY SHEET...
                for indx, _ in enumerate(df_sheet):
                    if sheet_name in ["Velocity", "Density", "Plans", "Flow"]:
                        df_sheet[indx] = np.where((df_sheet[indx] == 0), np.nan, df_sheet[indx])  # replace all 0s to nan
                    else:
                        df_sheet[indx] = np.where((df_sheet[indx] == np.inf) | (df_sheet[indx] == -np.inf), np.nan, df_sheet[indx])  # replace all infs to nan
                # df[indx][df[indx] == np.inf] = np.nan;            df[indx][df[indx] == -np.inf] = np.nan
                # find all ranges of upto maximum seq of NaNs allowed:
                range1, range2 = Find_Ranges(df_sheet, np.nan, max_NaNs)
                for indx, _ in enumerate(df_sheet):
                    for j in range(len(range1[indx])):
                        if sheet_name == "Plans":
                            if range1[indx][j] == 0:
                                continue
                            df_sheet[indx][range1[indx][j]:range2[indx][j]] = df_sheet[indx][range1[indx][j] - 1]  # same plan as the one before the missing seq
                        else:
                            # if j==4208:
                            #     print(str(j))
                            df_sheet[indx][range1[indx][j] - 1:range2[indx][j] + 1] = interpolate_array(df_sheet[indx][range1[indx][j] - 1:range2[indx][j] + 1])

            if (is_average):
                for indx, _ in enumerate(df_sheet):
                    df_sheet[indx] = np.correlate(df_sheet[indx], np.ones(avg_neighbors) / avg_neighbors, "same")
                    # although it perhaps remove non-sequential data, so maybe to use np.convolve: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy

            for indx, _ in enumerate(df_sheet):
                if sheet_name in ["Density", "Plans", "Flow"]:
                    df_sheet[indx] = np.where((df_sheet[indx] == np.nan), 0, df_sheet[indx])  # replace all nans back to 0s
            # finally update all this processing back to the data file:
            # for x in range(len(links_of_interest)):
            #    df[sheet_name][links_of_interest[x]] = df_sheet[x]
            df[sheet_name] = df_sheet
            indx += 1

    # df_info = df["Info"]
    # df_info = Assign_DF(df_info, "Post_process_"+str(sheet_name), [[perc_range, minrows_range, minrange_range, ranges, avg_neighbors],
    #                                             "max fraction of nans to count a row, max allowed nan rows,"+
    #                                             "min allowed range, output ranges, avg_neighbors"])
    # df["Info"] = df_info
    np.save(out_file, df)
    
    print("Finished Post_process")


def Post_process_Coding(out_file, nodes_file, sheet_name, code_file, sheet_code, sheet_group):
    # Process creating new encoded sheet from old sheet, using code table in sheet_code, grouping by table in sheet_group
    # E.g.: sheet_name = "Plans", sheet_code = "tblPrograms", sheet_group = "tblPrograms_group"
    new_sheet = sheet_name+"_coded"
    node_data = np.load(nodes_file, allow_pickle=True).item()
    node_data = node_data["Links_by_Name"]
    nodes_length = node_data.shape[0] # len(node_data[0])
    in_file = os.path.splitext(out_file)[0] + "_nan" + os.path.splitext(out_file)[1]
    filename = in_file
    if os.path.exists(out_file):
        filename = out_file
    #elif os.path.exists(in_file):

    df1 = np.load(filename, allow_pickle=True).item()
    df_name = df1[sheet_name]
    df_name2 = np.zeros(df_name.shape, dtype=np.int32)   # np.empty(df_name.shape, dtype=np.int32)
    #df_name2[:] = np.nan
    #  del df1; gc.collect()    --- cause i need to update it when saving
    df2 = np.load(code_file, allow_pickle=True).item()
    df_code = df2[sheet_code]
    df_group = df2[sheet_group]
    df_group[:, 1] = np.where(np.isnan(df_group[:, 1]), 0 ,df_group[:, 1]).astype(np.int32)
    #df_group1 = df_group.loc[:, 0].values
    #df_group2 = df_group.loc[:, 1].values
    #cols = df_code.columns
    #crossid = df_code[cols[0]].values
    #prognum = df_code[cols[1]].values
    #target = df_code[cols[3]].values
    #crossid = crossid[1:]   # remove the headers
    #prognum = prognum[1:]
    #target = target[1:]
    #total_rows = crossid.shape[0]
    df_code = df_code[1:,:]
    total_rows = df_code.shape[0]
    for index in range(total_rows):
#        if index==13:
        print('row '+str(index)+'/'+str(total_rows))
        if not(isint(df_code[index,1])) or (df_code[index,3]=='nan') or (np.isnan(df_code[index,3])):
            continue
# reasons for nan in Plan_Code table: nan from Plan, nan/empty target descr in Cross file, prognum is not int in Cross file.
        tar = df_group[:, 1][df_group[:, 0]==df_code[index,3]] + 10000  # +10000 to distinguish new replacements from old plan values
        #tar = int(df_code[index,0])*100 + int(df_code[index,1]) + 10000
        prog = int(df_code[index,1])
        curr_linkk = np.where(node_data[:,3] == int(df_code[index,0]))
        curr_linkk = np.squeeze(curr_linkk)
        curr_linkk = curr_linkk[(curr_linkk >= 0) & (curr_linkk < nodes_length)]
        # if find prog value of Plan, then change it to new 10000+code, otherwise keep previously changed plan->code cells:
        df_name2[:, curr_linkk] = np.where(df_name[:,curr_linkk] == prog, tar, df_name2[:,curr_linkk]) # , df_name[:,curr_linkk])

    df_name2 = np.where(df_name2 >= 10000, df_name2 - 10000, df_name2) # df_name2[df_name2>=10000] - 10000
    df1[new_sheet] = df_name2
    np.save(out_file, df1)
    print("Finished Post_process_Coding")


def RunRandom(nnseed):
    torch.manual_seed(nnseed)
    torch.cuda.manual_seed(nnseed)
    np.random.seed(nnseed)
    #random.seed(nnseed)
    torch.backends.cudnn.deterministic = True

def Permute_DataSets():
    # Either for different splittings of data (n_train, n_val, n_test) or random permutation of data, or both
    # NOTE: you need to change "perm" to have new random reordering..
    RunRandom(perm)
    # import all relevant files:
    out_file = data_fname_output
    out_file = os.path.splitext(out_file)[0] + "_06" + os.path.splitext(out_file)[1]  # special file, containing Data Sets additional info
    out_train = os.path.splitext(out_file)[0] + "_train" + os.path.splitext(out_file)[1]
    out_val = os.path.splitext(out_file)[0] + "_val" + os.path.splitext(out_file)[1]
    out_test = os.path.splitext(out_file)[0] + "_test" + os.path.splitext(out_file)[1]

    data_sets_x, data_sets_y = [], []
    datax, datay, indxs = np.load(out_train, allow_pickle=True).item().values()
    data_sets_x.append(datax.copy())
    data_sets_y.append(datay.copy())
    datax, datay = np.load(out_val, allow_pickle=True).item().values()
    data_sets_x.append(datax.copy())
    data_sets_y.append(datay.copy())
    datax, datay = np.load(out_test, allow_pickle=True).item().values()
    data_sets_x.append(datax.copy())
    data_sets_y.append(datay.copy())
    data_sets_x = np.concatenate(data_sets_x)  # dim = [#samples, input_channels,   #time steps, #nodes]
    data_sets_y = np.concatenate(data_sets_y)  # dim = [#samples, output_channels, #time steps, #nodes]
    tot_sets = len(data_sets_x)

    # data_sets_x = data_sets_x.tolist()  # turn into list the 1st dimension, so that we could index them
    # data_sets_y = data_sets_x.tolist()
    if perm > 0:  # change the order
        org_data = np.argsort(indxs, axis=None)  # restore original order
        data_sets_x = data_sets_x[org_data]
        data_sets_y = data_sets_y[org_data]
        indxs = np.linspace(0, tot_sets - 1, tot_sets, dtype=int)
        indxs = np.random.permutation(indxs)  # new random order
        data_sets_x = data_sets_x[indxs]
        data_sets_y = data_sets_y[indxs]

    nn_train = int(n_train / 100 * tot_sets)
    nn_val = int(n_val / 100 * tot_sets)
    nn_test = tot_sets - nn_train - nn_val
    # Store into files:
    np.save(out_train, {"x": data_sets_x[:nn_train, :, :, :], "y": data_sets_y[:nn_train, :, :, :], "indxs": indxs})  # only in train file store all indexes
    np.save(out_val, {"x": data_sets_x[nn_train:nn_train + nn_val, :, :, :], "y": data_sets_y[nn_train:nn_train + nn_val, :, :, :]})
    np.save(out_test, {"x": data_sets_x[nn_train + nn_val:, :, :, :], "y": data_sets_y[nn_train + nn_val:, :, :, :]})



def Post_DataSets():   # out_file, perm, in_sheets, out_sheets, n_train, n_val, n_test):
    # Create Datasets. I.e. tuples of (input, output) where the features are in "in" and "out" sheets;
    # Next: Divide into 3 files of train, val, test sets. Use perm: if -1 then the same order as in the file, otherwise
    # permutate by a seed.
    from sklearn.preprocessing import StandardScaler
    global curr_std, curr_dev

    in_sheets, out_sheets, out_file = input_sheets, output_sheets, data_fname_output
    # Define files: ------
    in_file = os.path.splitext(out_file)[0] + "_05" + os.path.splitext(out_file)[1]
    if not os.path.exists(in_file):  # if 04 don't exists - it's ok, it's not mandetory (it's optional)
        in_file = os.path.splitext(out_file)[0] + "_04" + os.path.splitext(out_file)[1]
    out_file = os.path.splitext(out_file)[0] + "_06" + os.path.splitext(out_file)[1]  # special file, containing Data Sets additional info
    out_train = os.path.splitext(out_file)[0] + "_train" + os.path.splitext(out_file)[1]
    out_val = os.path.splitext(out_file)[0] + "_val" + os.path.splitext(out_file)[1]
    out_test = os.path.splitext(out_file)[0] + "_test" + os.path.splitext(out_file)[1]
    df = np.load(in_file, allow_pickle=True).item()

    # Organize all features with time order: ------
    df1 = np.concatenate(df["StepsStart"])
    df2 = np.concatenate(df["StepsEnd"])
    df_links = np.zeros(df1.shape[0], dtype=int)  # stores all links indexes
    tot_links = len(df["StepsStart"])
    m = 0;   df_lens = []
    for indx in range(tot_links):
        n = df["StepsStart"][indx].shape[0]
        df_links[m:m + n]=indx
        m += n
        df_lens.append(m)
    cycs = df2 - df1  # array of cycles' length (already flatten, unlike: cycs = [df["StepsEnd"][i] - df["StepsStart"][i] for i in links_of_interest])
    #hist, bin_edges = np.histogram(cycs[(cycs>60) & (cycs<130)], density=False, bins=100)  # find most frequent range of cycle durations
    # UNHIDE THE BELOW if you want to see histogram of cycle duration distribution:
    # n, bins, patches = plt.hist(cycs[(cycs>75) & (cycs<105)], density=False, bins=1000)
    # plt.show()

    org_data = np.argsort(df1, axis=None)  # indexes of the sorted by time, FROM NOW we sort everything by time
    df1 = df1[org_data]
    del df2; gc.collect()
    cycs = cycs[org_data]
    df_links = df_links[org_data]

    # Join all relevant sheets into one multi-dimensional array (to be used to filter relevant data): ------
    sheets = in_sheets + out_sheets + ["ext"]   # output after input
    input_indexes, output_indexes, ext_indexes = np.split(np.linspace(0, len(sheets)-1, len(sheets), dtype=int), [len(in_sheets), len(in_sheets)+len(out_sheets)])
    input_indexes, output_indexes, ext_indexes = list(input_indexes), list(output_indexes), list(ext_indexes)
    df_all, continuous_indexes = [], []
    plan_indx = -1
    for indx, sheet in enumerate(sheets):
        print("Sheet: "+sheet)
        if sheet=="Plans":
            plan_indx = indx
        if sheet[:4] == "Time":  # add external features (not spatial or temporal)
            ltemp = len(df1)
            temp2 = np.empty(ltemp)
            for i in range(ltemp):
                tt = datetime.datetime.fromtimestamp(df1[i])
                if sheet == "Time_month":
                    temp2[i] = tt.month
                if sheet == "Time_day":
                    temp2[i] = tt.day
                if sheet == "Time_weekday":
                    temp2[i] = tt.weekday()
                if sheet == "Time_hour":
                    temp2[i] = tt.hour
                if sheet == "Time_cycle":
                    temp2[i] = cycs[i]    # cycle duration
            df_all.append(temp2.copy())
            continuous_indexes.append(indx)   # these are also continuous features...
        elif sheet=="ext":
            # additional feature(s), not used in modeling, only to store the start and end time of each data point
            df_all.append(np.linspace(0,len(df1)-1,len(df1), dtype=int))            
        else:
            temp = np.concatenate(df[sheet])[org_data] # organize each flatten sheet array by time:
            df_all.append(temp.copy())  # Every sheet dims~=[data*nodes]
            # Every sheet has its own code for missing-data, so we unite them here to one unique code = NaN:
            if sheet in ["Plans", "nVehs", "TravelTime", "Density", "Flow", "Velocity"]:  # all sheets where missing data = 0
                df_all[-1][df_all[-1] == 0] = np.nan
            if sheet in continuous_sheets:
                continuous_indexes.append(indx)
    df_all = np.stack(df_all).transpose()    # dim = [data*nodes, all_channels]
    max_Plans = -1
    if plan_indx > -1:
        max_Plans = max(df_all[:,plan_indx])

    # Gather all data point valid sequences: ------
    is_ok = "n"
    while (is_ok == "n"):
        temp1, temp2 = "", ""
        min_seq_length = 2
        # search for each feature alone, i.e. input features must have not-nan for them while output for theirs..
        # temp = input("Press maximum std (or default=0.2) and maximum deviation for start cycles (or default=20):")
        # if ' ' in temp:
        #     temp1, temp2 = temp.split()
        # else:
        #     temp1 = temp
        # if isfloat(temp1):
        #     max_std_cyc = float(temp1)
        # if isfloat(temp2):
        #     max_threshold = float(temp1)
        # temp = input("Enter minimum Total #Steps (=input+prediction), def=2:")
        # if isint(temp):
        #     min_seq_length = int(temp)
        # temp = input("Enter #Prediction Steps (<minimum), def=0=next step:")
        # if isint(temp):
        #     pred_steps = int(temp)
        range1, range2, steps = [], [], []
        temp_range1, temp_range2 = None, None
        avg_std, avg_dev = 0, 0
        curr_std, curr_dev = 0, 0
        temp_steps = 0
        i = tot_links
        meit = df1.shape[0]//10
        def check_i():
            global curr_std, curr_dev
            # check all conditions for current i, and return 2 if end of data, 0=true=conditions satisfied, 1=otherwise
            if i%meit==0:
                print("Searching for sequences of data: "+str(i+1)+"/"+str(df1.shape[0])+", found so far: "+str(len(steps)))
            if i >= df1.shape[0]:
                return 2
            temp = cycs[i - tot_links:i]
            temp_mean = np.mean(temp, axis=0)
            is_mean = True # abs(temp_mean-100)<5  # between 95 and 105
            curr_std = np.std(temp, axis=0) / temp_mean
            curr_dev = max(abs(df1[i - tot_links:i] - df1[i - 1]))
            is_features = np.any(np.isnan(df_all[i - tot_links:i, input_indexes]))
            if temp_steps > pred_steps:
                is_features = np.any(np.isnan(df_all[i - tot_links:i, output_indexes])) or is_features
            is_unique = len(np.unique(df_links[i - tot_links:i])) == tot_links  # check that all links are unique
            return int((is_mean) and (is_unique) and (not is_features) and (curr_dev < max_dev_threshold) and (curr_std < max_std_cyc))

        curr_check = check_i()
        while True:
            while curr_check==1:
                if temp_range1 is None:
                    temp_range1 = i - tot_links
                    temp_steps = 0
                temp_steps += 1
                avg_std += curr_std
                avg_dev += curr_dev
                temp_range2 = i  # actually it's i-1, but because of "range"...
                i += tot_links
                curr_check = check_i()
                if curr_check==2:
                    break
            if temp_range1 is not None:
                range1.append(temp_range1)
                range2.append(temp_range2)
                steps.append(temp_steps)
                temp_range1 = None  # no need to initialize temp_range2 and temp_steps, since we check only by temp_range1
            i += 1
            curr_check = check_i()
            if curr_check == 2:
                break

        # Here add filter out cycles with different length than the most frequent one:
        # ...

        # Print frequencies of the seqs, and asks if want to run with new parameters  ------
        avg_std /= np.sum(steps)
        avg_dev /= np.sum(steps)
        range1, range2, steps = np.array(range1), np.array(range2), np.array(steps)
        indxs = np.where(steps >= min_seq_length)[0]
        range1, range2, steps = range1[indxs], range2[indxs], steps[indxs]
        found_seqs, freqs = np.unique(steps, return_counts=True)
        print("(Steps,freq)=", [(found_seqs[i], freqs[i]) for i in range(len(found_seqs))])
        print("Std: Average: " + str(avg_std) + ", Current Max: " + str(max_std_cyc))
        print("Dev: Average: " + str(avg_dev) + ", Current Max: " + str(max_dev_threshold))
        is_ok = 'y' #input("OK to continue? (n=No)")

    # temp = input("Enter Total #Steps (=input+prediction), def=2:")
    # if isint(temp):
    #     seq_length = int(temp)
    # Pick only steps that are >= seq_length:
    indxs = np.where(steps >= seq_length)[0]
    range1 = range1[indxs]
    range2 = range2[indxs]
    steps = steps[indxs]

    # Finding all relevant data indexes and normalize them: ------
    input_seq = seq_length - pred_steps - 1
    all_data_sets = np.array([])
    #indexes_x, indexes_y = [], []
    for i in range(steps.shape[0]):   # block of data (a seq)
        for x in range(steps[i] - seq_length + 1):  # step in the seq
            indx = np.arange(range1[i] + x*tot_links, range1[i] + (x+input_seq)*tot_links)   # first seq-pred steps
            indy = np.arange(range1[i] + (x+seq_length-1)*tot_links, range1[i] + (x+seq_length)*tot_links)   # last step in the seq
            #indexes_x.append(indx);     indexes_y.append(indy);
            all_data_sets = np.unique(np.hstack([all_data_sets, indx, indy]))
    all_data_sets = all_data_sets.astype('int')
    # Normalization (on a non-missing data, both x's and y's), we do it for each feature alone:
    df["Scalers"] = []
    for i in continuous_indexes:
        scaler = StandardScaler()
        df_all[all_data_sets, i] = scaler.fit_transform(df_all[all_data_sets,i].reshape(-1, 1))[:, 0]
        df["Scalers"].append(scaler)


    # Create data sets:
    data_sets_x_point = np.zeros([len(input_indexes), input_seq, tot_links])  # [#input channels, #time steps, #nodes]
    data_sets_y_point = np.zeros([len(output_indexes+ext_indexes), 1, tot_links])  # [#output channels, #time steps, #nodes]
    data_sets_x = []
    data_sets_y = []
    for i in range(steps.shape[0]):   # block of data (a seq)
        for x in range(steps[i] - seq_length + 1):  # step in the seq
            for y in range(input_seq):  # UPDATED TO SORT FOR EACH seq (of tot_links items) seperatly..
                 indx = slice(range1[i] + (x+y)*tot_links, range1[i] + (x+y+1)*tot_links)
                 org_data2 = np.argsort(df_links[indx], axis=None)
                 data_sets_x_point[:,y,:] = df_all[indx,input_indexes][org_data2].transpose()
                     #.reshape((len(input_indexes), tot_links))  # dont need "+ext_indexes", since we never use it
            indy = slice(range1[i] + (x+seq_length-1)*tot_links, range1[i] + (x+seq_length)*tot_links)   # last step in the seq
            org_data2 = np.argsort(df_links[indy], axis=None)
            data_sets_y_point = np.expand_dims(df_all[indy,output_indexes+ext_indexes][org_data2].transpose(), axis=1)
                #.reshape((len(output_indexes+ext_indexes),1, tot_links))
            data_sets_x.append(data_sets_x_point.copy())
            data_sets_y.append(data_sets_y_point.copy())
    tot_sets = len(data_sets_x)


    # Divide into 3 files of train, val, test sets
    indxs = np.linspace(0, tot_sets - 1, tot_sets, dtype=int)  # the order of the new permutation of datasets, relative to the original order
    data_sets_x = np.array(data_sets_x)
    data_sets_y = np.array(data_sets_y)
    data_sets_x = np.stack(data_sets_x)  # dim = [#samples, input_channels,   #time steps, #nodes]
    data_sets_y = np.stack(data_sets_y)  # dim = [#samples, output_channels, #time steps, #nodes]
    nn_train = int(n_train / 100 * tot_sets)
    nn_val = int(n_val / 100 * tot_sets)
    nn_test = tot_sets - nn_train - nn_val
    # Store into files:
    np.save(out_train, {"x": data_sets_x[:nn_train, :, :, :], "y": data_sets_y[:nn_train, :, :, :], "indxs": indxs}) # only in train file store all indexes
    np.save(out_val, {"x": data_sets_x[nn_train:nn_train + nn_val, :, :, :], "y": data_sets_y[nn_train:nn_train + nn_val, :, :, :]})
    np.save(out_test, {"x": data_sets_x[nn_train + nn_val:, :, :, :], "y": data_sets_y[nn_train + nn_val:, :, :, :]})
    df_new = {"continuous_indexes": continuous_indexes, "sheets": sheets, "input_indexes": input_indexes,
              "output_indexes": output_indexes, "input_sheets": input_sheets, "output_sheets": output_sheets,
              "seq_length": seq_length, "pred_steps": pred_steps, "max_Plans": max_Plans,
              # "indexes_x": indexes_x, "indexes_y": indexes_y,
              "nn_train": nn_train, "nn_val": nn_val, "nn_test": nn_test, "tot_sets": tot_sets,
              "in_sheets": in_sheets, "out_sheets": out_sheets, "found_seqs": found_seqs, "freqs": freqs,
              "avg_std": avg_std, "avg_dev": avg_dev,
              "AdjMatrix": df["AdjMatrix"], "dict_names": df["dict_names"], "Info_Links": df["Info_Links"],
              "Info": df["Info"], "Scalers": df["Scalers"],
              "out_train": out_train, "out_val": out_val, "out_test": out_test}
    np.save(out_file, df_new)

    Permute_DataSets()

    print("Finished Post_Data_Prep")




def SmoothArray(df1, max_threshold):
    # Smooth close values in some array (e.g. cycle's start or length).

    # from "https://stackoverflow.com/questions/35136244/how-can-you-get-the-order-back-after-using-argsort":
    org_data = np.argsort(df1, axis=None)  # indexes of the sorted array
    df1 = np.sort(df1, axis=None)  # flat the array and sort it.
    diff = np.diff(df1)
    data0 = df1[0]
    del df1;  gc.collect()
    # Now we remove duplicates and close values:
    acc_diff = 0 # datetime.timedelta(0)
    previous_half_acc_diff = 0 # datetime.timedelta(0)
    first_indx = 0
    for indx, curr_diff in enumerate(diff):
        if acc_diff + curr_diff > max_threshold:
            if acc_diff > 0: # datetime.timedelta(0):  # store previous indexes
                if first_indx == 0:
                    diff[first_indx] = acc_diff / 2  # average of all what's accumulated
                else:
                    diff[first_indx] += acc_diff / 2 + previous_half_acc_diff # average of all what's accumulated
                previous_half_acc_diff = acc_diff / 2

                diff[first_indx + 1:indx] = 0 #datetime.timedelta(0)  # all rest = 0
                acc_diff = 0 # datetime.timedelta(0)  # reset accumulation
            first_indx = indx  # since no accmulation or ended we reset "first_indx"
        else:  # accumulate new difference
            acc_diff += curr_diff

    acc_diff = data0
    df1 = np.empty(diff.shape[0] + 1)#, dtype=float);
    df1.fill(0.0) # datetime.timedelta(0))
    df1[0] = acc_diff + diff[0]
    for indx, _ in enumerate(diff):
        acc_diff += diff[indx]
        df1[indx + 1] = acc_diff

    #del diff;  gc.collect()
    org_data = np.argsort(org_data, axis=None)  # sort back to original order
    df1 = df1[org_data]
    return df1 #org_data, df1


def Post_Data_SP():
    # Post processing of SP data (before applying BT transforming):
    # 1) Update "Plans" into one-hot vector indexes: using code table in sheet_code, grouping by table in sheet_group
    #    E.g.: sheet_name = "Plans", sheet_code = "tblPrograms", sheet_group = "tblPrograms_group"
    # 2) Change Steps by smoothing close values of starting cycles and of cycles' length to be the same
    # 3) if is_complete=True: add inner cycles where whole cycles can be filled, between disconnected cycles.

    out_file = data_fname_output

    in_file = os.path.splitext(out_file)[0] + "_02" + os.path.splitext(out_file)[1]
    out_file = os.path.splitext(out_file)[0] + "_03" + os.path.splitext(out_file)[1]
    df = np.load(in_file, allow_pickle=True).item()
    tot_links = len(df["StepsStart"])
    node_data = np.load(links_fname, allow_pickle=True).item()
    node_data = node_data["Links_by_Name"]
    nodes_length = node_data.shape[0]  # len(node_data[0])

    # Part 1: convert SP IDs specific to each intersection to a dictionary of IDs for all intersections:
    dfp = df["Plans"].copy() #  np.zeros(len(df["Plans"]), dtype=np.int32)  # np.empty(df_name.shape, dtype=np.int32)
    dfp = [np.full(i.shape,0) for i in dfp]  # fill all arrays with zeros
    # dfp[:] = np.nan
    #  del df; gc.collect()    --- cause i need to update it when saving
    df2 = np.load(code_fname_input, allow_pickle=True).item()
    df_code = df2[sheet_code]
    df_group = df2[sheet_group]
    df_group[:, 1] = np.where(pd.isna(df_group[:, 1]), 0, df_group[:, 1]).astype(np.int32)  # all nans turn to code=0

    df_code = df_code[1:, :]
    total_rows = df_code.shape[0]

    if SP_coding==0: # coding into groups based on description
        # go over all intersections and their plans:
        for index in range(total_rows):
            if index % 200 == 0:
                print('Plan coding: ' + str(index) + '/' + str(total_rows))
            if not (isint(df_code[index, 1])) or (df_code[index, 3] == 'nan') or (pd.isna(df_code[index, 3])):
                continue
            # reasons for nan in Plan_Code table: nan from Plan, nan/empty target descr in Cross file, prognum is not int in Cross file.
            tar = df_group[:, 1][df_group[:, 0] == df_code[
                index, 3]] + 100000  # +100000 to distinguish new replacements from old plan values
            # tar = int(df_code[index,0])*100 + int(df_code[index,1]) + 100000
            prog = int(df_code[index, 1])
            curr_linkk = np.where(node_data[:, 4] == int(df_code[index, 0]))[0]  # to intersection column
            curr_linkk = curr_linkk[(curr_linkk >= 0) & (curr_linkk < nodes_length)]
            # if find prog value of Plan, then change it to new 10000+code, otherwise keep previously changed plan->code cells:
            for cur_link in curr_linkk:
                dfp[cur_link] = np.where(df["Plans"][cur_link] == prog, tar, dfp[cur_link])  # , df_name[:,curr_linkk])
    elif SP_coding==1: # coding based on link+SP ID
        for index, _ in enumerate(dfp):
            dfp[index] = np.where(df["Plans"][index]>0, df["Plans"][index]+index*100+100000, df["Plans"][index])

    dfp = np.concatenate(dfp)  # for later use (it if is_complete=True)
    dfp = np.where(dfp >= 100000, dfp - 100000, dfp)  # dfp[dfp>=10000] - 10000
    #df["Plans"] = dfp  # cause i may use it if is_complete=True
    del df2, df_code, df_group; gc.collect()



    # Smooth cycles' startings:
    df1 = df["StepsStart"].copy()
    df2 = df["StepsEnd"].copy()
    m = 0;   df_lens = []
    for i in range(tot_links):
        m += df["StepsStart"][i].shape[0]
        df_lens.append(m)
    df1 = np.concatenate(df1)
    df2 = np.concatenate(df2)
    end_start = [True if df1[i] == df2[i - 1] else False for i in range(1, len(df1))]
    # NO NEED SMOOTHING START CYCLE TIMES, since: 1) the aggregating method isn't perfect (e.g. in cycle durations). 2) after completing new cycles
    # we get new starting times, so new smoothing is required. 3) Eventually it's unnecessary, in Dataset preperation we find them anyway..
    '''
    df1 = SmoothArray(df1, max_threshold)
    # copy ends from starts if they were the same before the smoothings above:
    df2 = np.array([df1[i] if end_start[i - 1] else df2[i - 1] for i in range(1, len(df1))] + [df2[-1]])
    '''

    # Smooth cycles' length:
    cycs = df2 - df1  # array of cycles' length (already flatten, unlike: cycs = [df["StepsEnd"][i] - df["StepsStart"][i] for i in links_of_interest])
    # FROM NOW ON IGNORE cycs, SINCE ITS AVERAGES MOVE AWAY CLOSE cycle times, e.g. 96->avg 92, 100->avg 102.5, so now instead difference=4, it's 10.5
    '''
    cycs = SmoothArray(cycs, max_threshold/2)

    #cycs = cycs[org_cycs]  # These are the new cycle lengths, so we update start and end cycle times:
    # Must first update end1, then start2, then end2, and so on (DONT DO IT, SINCE IT SHIFT ALL CYCLES..):
    ''
    df2[0] = df1[0] + cycs[0]
    for i in range(len(end_start)):
        if end_start[i]:
            df1[i+1] = df2[i]
        df2[i+1] = df1[i+1] + cycs[i+1]
    ''
    #df2 = df1 + cycs  # combine old starts with new cycle length thus produce new ends
    #  we go from beginning to end, for each new end we check if its connected to next start, thus update it, otherwise leave the old disconnected start as it was..
    #df1 = np.array([df1[0]] + [df2[i - 1] if end_start[i - 1] else df1[i] for i in range(1, len(df2))])
    '''


    dfg = np.concatenate(df["GreenTimes"])
    if is_complete:
        #diff = np.diff(df1)
        #last_valid = 0;   next_valid = 0
        df_lens = np.array(df_lens)
        df_lens2 = df_lens[:-1].copy()  # np.diff(df_lens)
        curr_link2 = len(df_lens)-2
        #end_start = list(end_start)  # already a list
        dff1 = np.empty(len(df1)*2, dtype=datetime.datetime)
        dff2 = np.empty(len(df1)*2, dtype=datetime.datetime)
        dffp = np.empty(len(df1)*2, dtype=np.uint16)  # uint8 for 0-255, but for different coding it may be more
        dffg = np.empty(len(df1)*2, dtype=np.int16)   # though new added greens are float
        #cycs = list(cycs)
        last_indx2 = dff1.shape[0] - 1
        meit = (len(df1) - 1) // 10
        last_indx = dff1.shape[0] - 1
        dff1[last_indx] = df1[df1.shape[0] - 1]
        dff2[last_indx] = df2[df1.shape[0] - 1]
        dffp[last_indx] = dfp[df1.shape[0] - 1]
        dffg[last_indx] = dfg[df1.shape[0] - 1]
        last_indx -= 1
        for i in reversed(range(df1.shape[0] - 1)):
            if (i+1==df_lens[curr_link2]):
                df_lens2[curr_link2] = last_indx+1
                #last_indx2 = last_indx
                curr_link2 -= 1
            if last_indx<100: # we must expand our arrays
                dff1 = np.insert(dff1, 0, np.empty(10000, dtype=datetime.datetime))
                dff2 = np.insert(dff1, 0, np.empty(10000, dtype=datetime.datetime))
                dffp = np.insert(dff1, 0, np.empty(10000, dtype=np.uint16))
                dffg = np.insert(dff1, 0, np.empty(10000, dtype=np.int16))
                last_indx += 10000
            if i % meit == 0:
                print('Complete cycles: ' + str(len(df1) - 2 - i) + '/' + str(len(df1) - 1))
            dfi1 = df1[i+1]
            end_start_diff = dfi1-df2[i]
            # if (not cycs[i + 1] == cycs[i]) and (abs(cycs[i + 1] - cycs[i])<max_threshold2):
            #     print(str(cycs[i])+","+str(cycs[i + 1]))
            # if dfi1==datetime.datetime.strptime("2019-07-01 09:36:38","%Y-%m-%d %H:%M:%S"):  # .%f
            #     print(str(i)+": "+ str(cycs[i]) + "," + str(cycs[i + 1]))
            if (end_start[i] == False) and (abs(cycs[i + 1] - cycs[i])<max_threshold2) and (end_start_diff >= cycs[i+1]):  # at least one cycle to fill the gap
                cyc_length = (cycs[i]+cycs[i + 1])/2.0      # average cycle
                new_cycles = int(end_start_diff // cyc_length)  # number of new cycles
                remainder = end_start_diff % cyc_length
                if remainder <= max_threshold:
                    pass
                elif cycs[i+1] - remainder <= max_threshold:
                    new_cycles += 1
                else:
                    dff1[last_indx] = df1[i]
                    dff2[last_indx] = df2[i]
                    dffp[last_indx] = dfp[i]
                    dffg[last_indx] = dfg[i]
                    last_indx -= 1
                    continue
                if new_cycles > max_cycles:
                    dff1[last_indx] = df1[i]
                    dff2[last_indx] = df2[i]
                    dffp[last_indx] = dfp[i]
                    dffg[last_indx] = dfg[i]
                    last_indx -= 1
                    continue
                cyc_length = end_start_diff/(new_cycles)  # equally spread the cycles between end and start of the gap, without changing original times
                df_lens[df_lens > i] += new_cycles  # increase all next indexes of link arrays by the same new appended amount
                dfgi1 = dfg[i + 1]
                efresh = (dfgi1 - dfg[i]) / (new_cycles + 1)  # we have new_cycles+1 jumps throughout new_cycles cycles
                dfgi1 -= efresh
                for x in reversed(range(new_cycles)):
                    # cycs.insert(i + 1 + x, cycs[i + 1])   # DONT NEED, CAUSE DONT USE CYCS
                    # Don't update df1 and df2 cause it need to be updated sequentially (changing one requires changing all next values):
                    #df1.insert(i + 1 + x, df2[i + x])
                    #df2.insert(i + 1 + x, df2[i + x] + cycs[i + 1])
                    # But add empty places for the new cycles:
                    dff1[last_indx] = dfi1-cyc_length
                    dff2[last_indx] = dfi1
                    dffp[last_indx] = dfp[i]
                    dffg[last_indx] = dfgi1
                    last_indx -= 1
                    dfi1 -= cyc_length
                    dfgi1 -= efresh
            dff1[last_indx] = df1[i]
            dff2[last_indx] = df2[i]
            dffp[last_indx] = dfp[i]
            dffg[last_indx] = dfg[i]
            last_indx -= 1
        df1 = dff1[last_indx+1:]
        df2 = dff2[last_indx+1:]
        dfp = dffp[last_indx+1:]
        dfg = dffg[last_indx+1:]
        '''
        # Again cycs changed, we update both df1 and df2. Must first update end1, then start2, then end2, and so on:
        # DONT CHANGE! SINCE IT SHIFT THE CYCLES, WON'T BE ABLE TO FIND SYNCHRONICITY LATER...
        df2[0] = df1[0] + cycs[0]
        for i in range(len(end_start)):
            if end_start[i]:
                df1[i + 1] = df2[i]
            df2[i + 1] = df1[i + 1] + cycs[i + 1]
        '''


    # break all flatten arrays back to their original form (list of different-size numpy arrays):
    df_lens2 = df_lens2 - last_indx - 1
    df1 = np.split(df1, df_lens2)   # df_lens[:-1]
    df2 = np.split(df2, df_lens2)
    dfp = np.split(dfp, df_lens2)
    dfg = np.split(dfg, df_lens2)

    # finally update all this processing back to the data file:
    df["StepsStart"] = df1  # Note: we smoothed Steps' starting cycle, but end's cycle stay unsmooth or copy the next start if it was the same before smoothing..
    df["StepsEnd"] = df2
    df["Plans"] = dfp
    df["GreenTimes"] = dfg
    np.save(out_file, df)
    print("Finished Post_Data_SP")



## Transformation Functions:



def TransformBT_Files():
    # To reduce processing time, we do this procedure each time we change "links_of_interest" array.
    # It used before any BT or SP processing, hence create only adjacency matrix and links list in df, and update BT files.

    out_file = data_fname_output

    node_data = np.load(links_fname, allow_pickle=True).item()["Links_by_Name"]
    node_name = node_data[:, 0]  # nodes = road links names
    node_length = node_data[:, 1]  # nodes = road links lengths
    node_move1 = node_data[:, 5]  # nodes = road links 1st move
    nodes_length = node_name.shape[0]

    links_of_interest = [88, 241, 248, 49, 173, 245, 251, 16, 34, 249, 256, 27, 116, 216, 252, 28, 117, 176, 262, 6,
                         260, 266, 12, 187, 263, 153, 254, 258, 24, 239]
    # links_of_interest = [2, 5, 24, 51, 66, 92, 104, 107, 152, 156, 159, 160, 161, 162, 164, 166, 167, 168]
    links_of_interest = [x - 2 for x in links_of_interest]

    # Part 1: remove all links but links_of_interest in df
    #in_file = os.path.splitext(out_file)[0] + "_0" + os.path.splitext(out_file)[1]
    out_file = os.path.splitext(out_file)[0] + "_01" + os.path.splitext(out_file)[1]
    df = {} # np.load(in_file, allow_pickle=True).item()  # contain tables/keys: "Info", "GreenTimes", "Plans", "StepsStart", "StepsEnd", "File1" (all from previous SP data transmission)
    tot_links = len(links_of_interest)
    df["AdjMatrix"] = np.zeros((tot_links, tot_links))  # adjacency matrix for links
    dict_names = {}   # translator of BT links to indexes of links_of_interest links
    dict_names2 = {}   # translator of SP links to indexes of links_of_interest links
    dict_names3 = []   # stores all "to" intersections (for SP files)
    for indx, i in enumerate(links_of_interest):
        # for BT files (links represented as pair of intersections):
        dict_names[node_name[i]] = indx  # row = 86: "TA152TA77" -> 86 -> 0
        # for SP files (links represented as pair of (to intersection, move):
        to_intersection = node_data[i, 4]
        if to_intersection not in dict_names3:
            dict_names3.append(to_intersection)
        for j in range(5, 10):   # columns: 4=to intersection, 5-9=moves
            if node_data[i, j] > 0:  # if there's a move
                if (to_intersection, node_data[i, j]) in dict_names2:
                    dict_names2[(to_intersection, node_data[i, j])].append(indx)  # sometimes same to intersection and move are for more than 1 link
                else:
                    dict_names2[(to_intersection, node_data[i, j])] = [indx]

    # Part 2: create BT files and reduce calculation in them
    regex = re.compile(".*\.csv")  # find all relevant BT data files in a folder
    # folder_content = os.listdir(folder_path) # Create the list of all your folder content with os.listdir()
    filenames = [os.path.join(root, file) for root, dirs, files in os.walk(csv_input_path) for file in files if regex.search(file)]
    gc.collect()

    for indx, fname in enumerate(filenames):
        BT = ConvertCsvtoNpy(fname, False)
        print("Start CSV File: " + fname + ", " + str(indx+1) + "/" + str(len(filenames)))

        len_data = BT.shape[0]
        if (BT[0, 0] == BT[0, 2]) or (len(BT[0])==22):  # sometimes there's additional redundant column.
            BT = np.delete(BT, 0, 1)  # remove 1st column of BT
        BT = np.delete(BT, slice(20, None), 1)
        BT = np.delete(BT, slice(11, 19), 1)  # 19->11
        valid = []
        gc.collect()

        for i in range(1, len_data):  # (1, len_data)
            if (pd.isna(BT[i, 4])) or (pd.isna(BT[i, 5])) or (not (BT[i, 11] == "Valid")):
                #BT[i,4] = -1 #BT = np.delete(BT, i, 0)  # if the value of entering/exiting node is NaN or if trip status isn't Valid then skip this row
                continue
            # assume only 1 item in curr_link array
            curr_link = dict_names.get(BT[i, 4] + BT[i, 5]) # search the indx of the link in nodes file, given enter and exit nodes.
            if curr_link==None:
                #BT[i,4] = -1 #BT = np.delete(BT, i, 0)  # if no such link, then we disregard this row
                continue

            valid.append(i)  # indexes of all Valid rows in BT file
            # This is for adjacent matrix (instead of 1's and 0's we accumulate appearances):
            if not (pd.isna(BT[i, 3])):
                link1 = dict_names.get(BT[i, 3] + BT[i, 4])
                if (not link1==None) and (node_length[link1] > 0) and (node_move1[link1] > 0):  # length>0 and move1>0
                    df["AdjMatrix"][link1, curr_link] += 1
            if not (pd.isna(BT[i, 6])):
                link3 = dict_names.get(BT[i, 5] + BT[i, 6])
                if (not link3==None) and (node_length[link3] > 0) and (node_move1[link3] > 0):
                    df["AdjMatrix"][curr_link, link3] += 1

            # turn all timestamps to datetime values to compare to df's cycle start/end times - NO, TIMESTAMPS ARE FASTER!
            # if BT[i, 8]>0:
            #     BT[i, 8] = datetime.datetime.fromtimestamp(BT[i, 8])
            # if BT[i, 10] > 0:
            #     BT[i, 10] = datetime.datetime.fromtimestamp(BT[i, 10])
            BT[i,4] = curr_link

        BT = BT[valid]
        BT = np.delete(BT, slice(11, None), 1)  # 11..
        BT = np.delete(BT, 9, 1)
        BT = np.delete(BT, slice(5,8), 1)       # 5-7
        BT = np.delete(BT, slice(4), 1)         # 0-3
        fname = os.path.splitext(fname)[0] + ".npy"
        np.save(fname, BT)
        del BT;  gc.collect()

    df["dict_names"] = dict_names
    df["dict_names2"] = dict_names2
    df["dict_names3"] = dict_names3
    df["Info_Links"] = links_of_interest
    np.save(out_file, df)
    print('Finished TransformBT_Files!')


def TransformBT():
    # Transform BT link single-vehicle travel times into time average sequence graph of links.
    # Produce all relevant tables: velocity, #vehs (flow for each time step), optional: density, avg. travel time.
    # Since it comes after importing SP data and processing it, we don't need to be given days and dates as in SP procedure.
    # Also Normalize all continuous sheets.. (here because all sheets are introduced only here, not in SP)

    out_file = data_fname_output

    in_file = os.path.splitext(out_file)[0] + "_04" + os.path.splitext(out_file)[1]
    if not os.path.exists(in_file):     # if 04 don't exists - it's ok, it's not mandetory (it's optional)
        in_file = os.path.splitext(out_file)[0] + "_03" + os.path.splitext(out_file)[1]
    out_file = os.path.splitext(out_file)[0] + "_04" + os.path.splitext(out_file)[1]
    # links file (or "nodes" in a graph):
    node_data = np.load(links_fname, allow_pickle=True).item()
    node_name = node_data["Links_by_Name"][:,0]    # nodes = road links names
    node_length = node_data["Links_by_Name"][:,1]  # nodes = road links lengths
    node_move1 = node_data["Links_by_Name"][:,5]  # nodes = road links 1st move
    nodes_length = node_name.shape[0]

    # Input files are: Data nan file assuming after SP data were inserted in it:
    df = np.load(in_file, allow_pickle=True).item()  # contain tables/keys: "Info", "GreenTimes", "Plans", "StepsStart", "StepsEnd", "File1" (all from previous SP data transmission)
    node_length = node_length[df["Info_Links"]]  # hold length only of selected nodes
    tot_links = len(df["StepsStart"])
    # if there're new sheets in df - initialize them
    for sheet in BT_sheets:
        if sheet not in df:
            df[sheet] = []
            for i in range(tot_links):
                l = df["StepsStart"][i].shape[0]  # size of each link's array
                df[sheet].append(np.zeros(l))
                # df["Velocity"][i].fill(np.nan)  # None  DONT DO IT SINCE WE ACCUMULATE IT FROM ZERO

    # for i in range(df["AdjMatrix"].shape[0]):
    #     df["Velocity"][i] = np.divide(np.ones_like(df["Velocity"][i]),df["Velocity"][i])  # df["Velocity"][i] /= df["nVehs"][i]
    #     df["Velocity"][i] = np.divide(df["Velocity"][i],np.power(df["nVehs"][i],2))  #
    # np.save(out_file, df)

    regex = re.compile(".*\.npy")  #(".*\.csv")  # find all relevant BT data files in a folder
    # folder_content = os.listdir(folder_path) # Create the list of all your folder content with os.listdir()
    filenames = [os.path.join(root, file) for root, dirs, files in os.walk(csv_input_path) for file in files if regex.search(file)]

    for indx, fname in enumerate(filenames):
        BT = np.load(fname, allow_pickle=True)   # = ConvertCsvtoNpy(fname, False)
        print("Start NPY File: " + fname + ", " + str(indx+1) + "/" + str(len(filenames)))
        len_data = BT.shape[0]
        last_search = np.full(tot_links, 1000)
        for i in range(1, len_data):  # (1, len_data)
            # assume only 1 item in curr_link array
            curr_link = BT[i,0] #dict_names.get(BT[i, 4] + BT[i, 5])
            # if curr_link==-1:
            #     continue

            # find the first cycle, that its start<entrance time of the veh
            bt_point1 = BT[i, 1]
            bt_point2 = BT[i, 2]
            last_search[curr_link] -= 1000
            max_len = df["StepsStart"][curr_link].shape[0]
            while (last_search[curr_link]<max_len) and (df["StepsStart"][curr_link][last_search[curr_link]]<=bt_point1):
                last_search[curr_link] += 1
            if last_search[curr_link]==max_len:
                continue
            cycle1 = last_search[curr_link]-1
            if last_search[curr_link]<1000:
                last_search[curr_link] += 1000
            #cycle1 = np.squeeze(np.where(df["StepsStart"][curr_link] <= bt_point1)[0])[-1]  # instead of max, since it's sorted
            curr_vel = node_length[curr_link] / (bt_point2 - bt_point1)  # calculated regardless to cycles, but from entring to exiting the link.
            x = cycle1
            # if x==82169:
            #     print(x)
            curr_start = df["StepsStart"][curr_link][x]  # current cycle start
            while (bt_point2 > curr_start): # and (prev_end==new_start):
                curr_end = df["StepsEnd"][curr_link][x]  # current cycle end
                curr_diff = min(curr_end,bt_point2) - max(curr_start,bt_point1)
                if curr_diff>0: #datetime.timedelta(0):
                    df["TravelTime"][curr_link][x] = curr_diff  #.seconds
                    df["nVehs"][curr_link][x] += 1
                    df["Velocity"][curr_link][x] += curr_vel
                x += 1
                if x>=max_len:  # reached the end of this array
                    break
                curr_start = df["StepsStart"][curr_link][x]  # new current cycle start (for the next loop step)

        del BT;  gc.collect()
        np.save(out_file, df)   # each BT file end of process - we update the output file


    # Now remove all links (rows and columns) where data frequency is less then some threshold:
    '''
    thr_freq = np.sum(df["AdjMatrix"]) * 1 / 100 / nodes_length  # the threshold for "no-data" = 1%
    for x in reversed(range(df["AdjMatrix"].shape[0])):
        if (np.sum(df["AdjMatrix"][:, x] + df["AdjMatrix"][x, :]) < thr_freq):  # or (x not in links_of_interest):
            df["AdjMatrix"] = np.delete(df["AdjMatrix"], x, 0)
            df["AdjMatrix"] = np.delete(df["AdjMatrix"], x, 1)
            df["Info_Links"].pop(x)
            df["StepsStart"].pop(x)
            df["StepsEnd"].pop(x)
            df["Plans"].pop(x)
            df["GreenTimes"].pop(x)
            df["Velocity"].pop(x)
            df["nVehs"].pop(x)
            df["TravelTime"].pop(x)
            df["Density"].pop(x)
            df["Flow"].pop(x)
    '''
    df["Info"]["Feature"].append("CSV files")
    df["Info"]["Value"].append(filenames)
    df["Info"]["Description"].append("")

    for i in range(df["AdjMatrix"].shape[0]):
        df["TravelTime"][i] = np.divide(df["TravelTime"][i],df["nVehs"][i])  # df["TravelTime"][i] /= df["nVehs"][i]
        df["Velocity"][i] = np.divide(df["Velocity"][i],df["nVehs"][i])  # df["Velocity"][i] /= df["nVehs"][i]
        df["Density"][i] = np.divide(df["nVehs"][i],node_length[i])  # df["nVehs"][i]/node_length[i]
        diff_cycle = df["StepsEnd"][i] - df["StepsStart"][i] #np.diff(df["StepsStart"][i])
        # for j in range(diff_cycle.shape[0]):
        #     diff_cycle[j] = diff_cycle[j].seconds
        df["Flow"][i] = np.divide(df["nVehs"][i],diff_cycle)  # df["nVehs"][i]/diff_cycle



    # We assume that there's order: 1st SP data, then BT data. So "Info" defined in SP transmission for the first time here.
    # Note we actually have no info for BT, as we don't had in SP..

    # Normalization (CHANGE IT TO NORMALIZE OVER THE FLATTEN ARRAY!!):
    # CANT DO IT SINCE THERE ARE MISSING DATA... ONLY ON FULL DATA BLOCK DO IT!


    # Store this file also (to replicate results):
    f = open(sys.argv[0], mode='rb')
    df["File2"] = f.read()     # File1 is for SP transmission, File2 is for BT.
    f.close()
    np.save(out_file, df)

    print('Finished TransformBT!')




def UnionOfTimes(green_data,green_part1, initial_greens, linkss, last_start_cycle):
    # Based on these:
    # https://stackoverflow.com/questions/26310046/calculating-total-time-when-given-ranges-of-time-that-may-overlap
    # https://stackoverflow.com/questions/11480031/merging-overlapping-time-intervals
    # Create tuples of ranges (start, end):
    ranges = []
    for l in linkss:
        ranges.append((green_data[l,3],green_data[l,3]+green_part1[l-last_start_cycle]*datetime.timedelta(seconds=1)))
        if (initial_greens is not None) and (initial_greens[l - last_start_cycle]>0):
            ranges.append((green_data[last_start_cycle-1,1], green_data[last_start_cycle-1,1] + initial_greens[l - last_start_cycle] * datetime.timedelta(seconds=1)))
    ranges.sort()  # Sort list of tuples by their first item
    ranges = [list(elem) for elem in ranges]  # turn this to list of lists instead of list of tuples (since we cannot update them)
    continuous_range = ranges[0]
    tot_green = 0
    for st,en in ranges[1:]:
        if st <= continuous_range[1]:
            continuous_range[1] = max(continuous_range[1], en)  # expand the continuous range
        else: # start a new seperated range
            tot_green += (continuous_range[1] - continuous_range[0]).seconds  # accumulate last continuous range
            continuous_range[0] = st
            continuous_range[1] = en
    tot_green += (continuous_range[1] - continuous_range[0]).seconds  # accumulate last continuous range
    return tot_green



def TransformSP():
    # Transform green durations into sum of them over some period, and traffic signal plans into most dominant plan over some period of time.
    # Only unlike "TransformBT_graph4", we don't have fix time steps sizes, but according to cycle time, AND we saved it into NPY (not EXCEL)!

    out_file = data_fname_output

    in_file = os.path.splitext(out_file)[0] + "_01" + os.path.splitext(out_file)[1]
    out_file = os.path.splitext(out_file)[0] + "_02" + os.path.splitext(out_file)[1]
    # links file (or "nodes" in a graph):
    node_data = np.load(links_fname, allow_pickle=True).item()
    node_data = node_data["Links_by_Name"]
    df = np.load(in_file, allow_pickle=True).item()
    tot_links = len(df["Info_Links"]) # node_data.shape[0]
    dict_names2 = df["dict_names2"]
    dict_names3 = df["dict_names3"]
    # Input file:
    # green_data = ConvertSQLGreenstoNpy()
    # Data: green_data[data points][0=#intersection, 1=cyc time stamp, 2=#move, 3=start green time, 4=green duration, 5=#prog].
    #df = {}
    for sheet in SP_sheets:
        df[sheet] = [[] for _ in range(tot_links)]
    last_start_cycle = 0
    file_number = 1
    nmoves = 0
    nfiles = 0

    while os.path.exists(SP_Files + "_" + str(file_number) + ".npy"):
        green_data = np.load(SP_Files + "_" + str(file_number) + ".npy", allow_pickle = True)
        nfiles += 1
        print("SP File number: "+str(nfiles))
        green_data = green_data.transpose()
        len_data = green_data.shape[0]

        for i in range(0, len_data):  # (1, len_data)   14900000+40985=78 intersection, 9668000=36
            curr_date = green_data[i,3].date()
            # if i>=15154200:
            #     print(i)
            # look only on the relevant data:
            if not ((green_data[i,0] in dict_names3) and (green_data[i,3].weekday() in weekdays) and (curr_date >= start) and (curr_date <= finish)):  # and isint(green_data[i, 2])):
                continue
            diff_cycle = (green_data[i,1] - green_data[i-1,1]).seconds
            if diff_cycle > 0:  # new cycle
                old_nmoves = nmoves
                nmoves = i - last_start_cycle  # number of moves in the last cycle
                # if start of green time is earlier then the end of the previous cycle
                diff_cycle2 = (green_data[last_start_cycle,1] - green_data[last_start_cycle - 1,1]).seconds
                if (not old_nmoves == nmoves) or (any([(green_data[last_start_cycle + x,3] < green_data[last_start_cycle - 1,1]) or
                                                       (green_data[last_start_cycle + x,4]<0) or
                                                       (green_data[last_start_cycle + x,3]>=green_data[last_start_cycle + x,1]) for x in
                         range(nmoves)])) or (diff_cycle2 < 60) or (diff_cycle2 > 130):
                    last_start_cycle = i
                    initial_greens = None  # erase previous array of adding greens, also do it when new intersection encountered.
                    continue
                green_part2 = green_data[last_start_cycle:i,3] + green_data[last_start_cycle:i,4] * datetime.timedelta(seconds=1) - green_data[last_start_cycle:i,1]
                # = green_data[last_start_cycle-1,1]+green_data[last_start_cycle:i,4]*datetime.timedelta(seconds=1) - green_data[last_start_cycle:i,1]
                for indx, x in enumerate(green_part2):
                    if x < datetime.timedelta(0):
                        green_part2[indx] = 0
                    else:
                        green_part2[indx] = x.seconds
                green_part1 = green_data[last_start_cycle:i,4] - green_part2
                if initial_greens is not None:
                    if (all(initial_greens == 0)) or (not (len(initial_greens) == len(green_part1))):
                        initial_greens = None
                # Gather all different links (for the case where we have multiple moves for one link):
                curr_links = {}
                nnode = green_data[last_start_cycle,0]  # it doesn't change during a cycle
                for x in range(last_start_cycle, i):
                    if not isint(green_data[x,2]):
                        continue
                    movee = (nnode, int(green_data[x,2]))  # a tuple of #intersection and #move
                    if movee in dict_names2:
                        links = dict_names2[movee]
                        for linkk in links:
                            if linkk in curr_links:
                                curr_links[linkk].append(x)
                            else:
                                curr_links[linkk] = [x]

                for linkk in curr_links.keys():
                    # if (linkk==1) and (green_data[last_start_cycle - 1,1].hour==16) and (green_data[last_start_cycle - 1,1].date()==datetime.date(2019, 11, 5)):
                    #     print(linkk)
                    df["StepsStart"][linkk].append(green_data[last_start_cycle - 1,1])  # store start of this cycle (not the end!)
                    df["StepsEnd"][linkk].append(green_data[last_start_cycle,1])  # store end of this cycle
                    df["Plans"][linkk].append(green_data[i - 1,5])  # store the plan in this cycle (assuming not changing in it)
                    # for movee in curr_links[linkk]:
                    if len(curr_links[linkk]) == 1:
                        x = curr_links[linkk][0] - last_start_cycle
                        tot_green = green_part1[x]
                        if initial_greens is not None:
                            tot_green += initial_greens[x]  # assuming they're not overlap
                    else:
                        tot_green = UnionOfTimes(green_data, green_part1, initial_greens, curr_links[linkk], last_start_cycle)
                    df["GreenTimes"][linkk].append(tot_green)

                if (not (all(green_part2 == 0))):
                    initial_greens = green_part2  # for next cycle
                last_start_cycle = i
        del green_data
        file_number += 1
        gc.collect()

    if "Info" not in df:
        df["Info"] = {}    # We assume that there's order: 1st SP data, then BT data. So we introduce "Info" for the first time here. Note we actually have no info for SP..
    df["Info"]["Feature"] = []
    df["Info"]["Value"] = []
    df["Info"]["Description"] = []
    # Replace inner lists with numpy arrays (faster and less memory):
    df["GreenTimes"] = [np.array(x) for x in df["GreenTimes"]]
    df["Plans"] = [np.array(x) for x in df["Plans"]]
    df["StepsStart"] = [np.array(x) for x in df["StepsStart"]]
    df["StepsEnd"] = [np.array(x) for x in df["StepsEnd"]]
    for i in range(tot_links):
        l = df["StepsStart"][i].shape[0]  # size of each link's array
        for j in range(l):
            df["StepsStart"][i][j] = datetime.datetime.timestamp(df["StepsStart"][i][j])
            df["StepsEnd"][i][j] = datetime.datetime.timestamp(df["StepsEnd"][i][j])
    # Store this file also (to replicate results):
    f = open(sys.argv[0], mode='rb')
    df["File1"] = f.read()
    f.close()
    np.save(out_file, df)

    print('Finished TransformSP!')





## Conversion Functions:

def ConvertSQLGreenstoNpy():
    gc.collect()
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                          'Server=TSMART-7;'
                          'Database=CommonDB;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    query = 'SELECT * FROM CommonDB.dbo.tblHISTORIC_GREEN_DURATION'
    cursor.execute(query)
    #rows = np.array(cursor.fetchall())
    return np.array(cursor.fetchall())
    #np.save(os.path.join(csv_input_path,'GreenDurations_'+time.strftime("%Y%m%d-%H%M%S")+'.npy'),rows)
    #np.savez_compressed(os.path.join(csv_input_path,'GreenDurations_'+time.strftime("%Y%m%d-%H%M%S")+'.npz'),rows)

def ConvertExceltoNpy(inp_file, is_header=False):
    out_file = os.path.splitext(inp_file)[0]# + ".npy"
    inp_file = os.path.splitext(inp_file)[0] + ".xlsx"
    if is_header:
        head = 'infer'
    else:
        head = None
    df = {}
    if os.path.exists(inp_file):
        #book = load_workbook(inp_file)
        #for ws in book.worksheets:
        book = pd.read_excel(inp_file, sheet_name=None, header=head, index_col=None) # If sheet_name=None: all sheets are returned
        for ws in book:
            df[ws] = book[ws].values
        np.save(out_file, df)

def ConvertCsvtoNpy(inp_file, is_save, is_header=False):
    out_file = os.path.splitext(inp_file)[0]# + ".npy"
    inp_file = os.path.splitext(inp_file)[0] + ".csv"
    if is_header:
        head = 'infer'
    else:
        head = None
    df = {}
    if os.path.exists(inp_file):
        try:
            df = pd.read_csv(inp_file, header=head, encoding='utf-8').values
        except:
            try:
                df = pd.read_csv(inp_file, header=head, encoding='cp1252').values
            except:
                try:
                    df = pd.read_csv(inp_file, header=head, encoding='ISO-8859-1').values
                except:
                    pass
        if is_save:
            np.save(out_file, df)
    return df

def ConvertNpytoExcel(inp_file, is_header=False):
    out_file = os.path.splitext(inp_file)[0] + "b.xlsx"
    inp_file = os.path.splitext(inp_file)[0] + ".npy"
    if is_header:
        head = 'infer'
    else:
        head = None
    if os.path.exists(inp_file):
        container = np.load(inp_file, allow_pickle=True).item()
        #data = [container[key] for key in container]
        writer = pd.ExcelWriter(out_file, engine='openpyxl')
        for key in container:
            pd.DataFrame(container[key]).to_excel(writer, sheet_name=key, header=head, index=None)
        writer.save()
        writer.close()



##-------------## Analyze data

def CompareChannels(inp_file, output_file, nnode, nrange):
    # compare in plot, all channels on the same graph: Velocity, GreenTimes, nVehs, ...
    from matplotlib import pyplot as plt
    fig = plt.figure(dpi=600)
    Font_Size = 14
    lw = 1.0 # linewidth
    plt.xlabel("5-min Steps", fontsize=Font_Size)
    plt.ylabel("Values", fontsize=Font_Size)
    container = np.load(inp_file, allow_pickle=True).item()
    for key in container:
        if key=="Info":
            continue
        df = container[key][nrange,nnode]
        df[np.isnan(df)]=0
        if key == "GreenTimes":
            df = df/6
        if key == "nVehs":
            lw = 0.5
        plt.plot(range(1, len(df) + 1), df, label=key,linewidth=lw)  # + args_length])
        lw = 1.0

    # for plot location choose: best, upper right, upper left, lower left, lower right,
    # 	right, center left,	center right, lower center,	upper center, center
    plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.9), prop={'size': Font_Size-3})
    plt.xticks(fontsize=Font_Size)
    plt.yticks(fontsize=Font_Size)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 22))

    if pd.notna(output_file):
        plt.savefig(output_file)

    plt.show()   # always show after saving!

    print('Finished CompareChannels!')


def Check_Data(out_file, sheet_name, nrange, fig_name):
    # Check how much missing data (sum) is in some range of steps, in the "nan" data file, for all sheets in sheet_name.
    out_file = os.path.splitext(out_file)[0] + "_nan" + os.path.splitext(out_file)[1]
    from matplotlib import pyplot as plt
    fig = plt.figure(dpi=600)
    Font_Size = 14
    lw = 1.0 # linewidth
    plt.xlabel("5-min Steps", fontsize=Font_Size)
    plt.ylabel("Values", fontsize=Font_Size)
    container = np.load(out_file, allow_pickle=True).item()
    nrange = np.arange(nrange[0],nrange[1])
    n_route = container[sheet_name[0]].shape[1]   # number of nodes
    MSEMAP = np.zeros((len(sheet_name)+1, n_route))
    for node in range(n_route):
        for indx, key in enumerate(sheet_name):
            if key == "Info":
                continue
            sum_missing = np.sum(container[key][nrange, node]== -np.inf) + \
                          np.sum(container[key][nrange, node]== np.inf) + np.sum(np.isnan(container[key][nrange, node]))
            MSEMAP[indx, node] = sum_missing / len(nrange) * 100  # percentage from total data
            # if key == "GreenTimes":
            #     df = df / 6
            # if key == "nVehs":
            #     lw = 0.5
    MSEMAP[len(sheet_name),:] = np.sum(MSEMAP[0:len(sheet_name),:], axis=0)
    for i in range(len(sheet_name)+1):
        if i<len(sheet_name):
            sn = sheet_name[i]
        else:
            sn = "total"
            plt.plot(range(1, n_route + 1), MSEMAP[i], label=sn, linewidth=lw)  # + args_length])
    lw = 1.0

    # for plot location choose: best, upper right, upper left, lower left, lower right,
    # 	right, center left,	center right, lower center,	upper center, center
    plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.9), prop={'size': Font_Size-3})
    plt.xticks(fontsize=Font_Size)
    plt.yticks(fontsize=Font_Size)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 22))

    if pd.notna(fig_name):
        plt.savefig(fig_name)

    plt.show()   # always show after saving!

    print('Finished Check_Data!')


def timestamp2(dt):
    #print(dt)
    return (dt - datetime.datetime(1970, 1, 1))/datetime.timedelta(seconds=1) #.total_seconds()

def PeriodicityGreens(green_data, out_file, days, time_start, time_end, start_date, end_date):
    # Plot green duration for all moves, in every intersection, for some days in a week (range), daily (over 24 hours).
    # Data: green_data[data points][0=#intersection, 1=cyc time stamp, 2=#move, 3=start green time, 4=green duration, 5=#prog].
    '''
    if os.path.exists(inp_file):
        with open(inp_file) as f:
            ff = next(f)
        fp = open(inp_file, "r")
        line = fp.readline()
        #fp = np.memmap(inp_file, mode='r')
        container = np.load(inp_file, mmap_mode='r')#.item()
        a = 0
    '''
    # prepare for plot:
    Font_Size = 12
    fig = plt.figure(dpi=1200)
    # plt.xlabel("Time in a 24hr day", fontsize=Font_Size)
    # plt.ylabel("MoveNum", fontsize=Font_Size)
    curr_intersection = green_data[0][0]
    #curr_date = green_data[0][3].date()
    initial_key = timestamp2(datetime.datetime(green_data[0][3].year, green_data[0][3].month, green_data[0][3].day))
    colors = ['red', 'green', 'blue']
    start_arr = {}
    finish_arr = {}
    move_arr = {}
    for green_row in green_data:
        curr_date = green_row[3].date()
        curr_key = str(curr_date)
        if (green_row[3].weekday() in days) and (curr_date>=start_date) and (curr_date<=end_date):
            if green_row[0] == curr_intersection:
                # if not (green_row[3].date() == curr_date):
                #     start_arr.append([])
                #     finish_arr.append([])
                #     move_arr.append([])
                #     curr_date = green_row[3].date()
                if not (curr_key in start_arr.keys()):
                    start_arr[curr_key] = []
                    finish_arr[curr_key] = []
                    move_arr[curr_key] = []
                curr_time = green_row[3].hour + green_row[3].minute / 60.0 + green_row[3].second / 3600.0
                if (curr_time>=time_start) and (curr_time<=time_end) and (isint(green_row[2])):
                    start_arr[curr_key].append(curr_time)
                    finish_arr[curr_key].append(curr_time + green_row[4] / 3600.0)
                    move_arr[curr_key].append(int(green_row[2]))
                    # curr_duration = green_row[4].second/3600.0
            else:
                plt.clf() # clear previous content
                plt.cla()
                plt.close(fig)
                plt.close('all')
                gc.collect()
                fig = plt.figure(dpi=1200)
                ax = plt.gca() #plt.subplot(1, 1, 1)
                plt.xlabel("Time in a 24hr day", fontsize=Font_Size)
                plt.ylabel("MoveNum", fontsize=Font_Size)
                for keyy in move_arr.keys(): # seperated keys, by days
                    curr_tstamp = timestamp2(datetime.datetime.strptime(keyy, '%Y-%m-%d'))
                    rel_day = (curr_tstamp - initial_key) / 86400.0
                    plt.hlines(np.array(move_arr[keyy])+rel_day/100.0, start_arr[keyy], finish_arr[keyy], linewidth=0.3, colors=colors[int(rel_day)%3])   # range(1, len(time_arr) + 1)
                plt.minorticks_on()
                plt.xticks(fontsize=Font_Size)
                plt.yticks(fontsize=Font_Size)
                ax.xaxis.set_major_locator(ticker.MultipleLocator((time_end-time_start)/5.0))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator((time_end-time_start)/50.0))
                # Show the major grid lines with dark grey lines
                plt.grid(axis = 'x', b=True, which='major', color='#666666', linewidth=0.2, linestyle='-')
                # Show the minor grid lines with very faint and almost transparent grey lines
                plt.grid(axis = 'x', b=True, which='minor', color='#666666', linewidth=0.1, linestyle='-', alpha=0.9)

                fname = out_file+'_'+str(time_start)+'_'+str(time_end)+'_'+str(curr_intersection)+'.png'
                # os.chdir(folder_path)
                if not os.path.exists(fname):
                    plt.savefig(fname)
                #plt.show()
                # new arrays and insert new data to them:
                start_arr = {}
                finish_arr = {}
                move_arr = {}
                curr_intersection = green_row[0]
                if not (curr_key in start_arr.keys()):
                    start_arr[curr_key] = []
                    finish_arr[curr_key] = []
                    move_arr[curr_key] = []
                curr_time = green_row[3].hour + green_row[3].minute / 60.0 + green_row[3].second / 3600.0
                if (curr_time>=time_start) and (curr_time<=time_end) and (isint(green_row[2])):
                    start_arr[curr_key].append(curr_time)
                    finish_arr[curr_key].append(curr_time + green_row[4] / 3600.0)
                    move_arr[curr_key].append(int(green_row[2]))

    print('Finished PeriodicityGreens!')


def PeriodicityGreens3(green_data, out_file, days, time_start, time_end, start_date, end_date):
    # Save an array of data where previous fix #steps have constant cycle time but varying green durations
    # Data: green_data[data points][0=#intersection, 1=cyc time stamp, 2=#move, 3=start green time, 4=green duration, 5=#prog].
    curr_intersection = green_data[0][0]
    # curr_date = green_data[0][3].date()
    range_indxs = []  # store all found good ranges of data
    cmap = plt.get_cmap('tab20')     # see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    len_data = green_data.shape[0]
    prev_steps = 9  # the fixed length (sliding array over the data)
    detected = False  # true if detected
    max_std_cyc = 0.02  # maximum allowable std/mean for the previous cyc time stamps
    min_std_green = 0.5  # minimum allowable std/mean for the previous duration times
    Font_Size = 12
    cyc_durations = []
    start_cycle_indx = [0]

    for i in range(1, len_data):  # (1, len_data)
        curr_date = green_data[i, 3].date()
        # look only on the relevant data:
        if (green_data[i, 3].weekday() in days) and (curr_date >= start_date) and (curr_date <= end_date) and isint(green_data[i, 2]):
            diff_cycle = (green_data[i][1]-green_data[i-1][1]).seconds
            if diff_cycle > 0: # new cycle
                cyc_durations.append(diff_cycle)
                start_cycle_indx.append(i)  # index of new cycle
                # check previous cycles, if they are similar (not have to be identical)
                cyc_num = np.array(cyc_durations[-prev_steps:])
                if any(cyc_num > 130.0) or any(cyc_num < 60.0) or len(cyc_num)<prev_steps:  #
                    continue
                if np.std(cyc_num, axis=0) / np.mean(cyc_num, axis=0) > max_std_cyc:  # if cycles duration not similar
                    continue
                nmoves = list(set(np.diff(start_cycle_indx[-prev_steps:])))
                if not(len(nmoves)==1): # if there're not the same amount of moves
                    continue
                nmoves = nmoves[0]
                data_block = green_data[start_cycle_indx[-prev_steps]:i,:].reshape((prev_steps-1,nmoves,6))
                #if not (len(np.unique(data_block[:,:,5]))==1):   # if all programs aren;t the same
                    # continue
                #if len(set(green_data[start_cycle_indx[-prev_steps]:i-1, 5]))==0: # if all signal plans are identical
                tot_deviance = 0
                for k in range(nmoves):
                    tot_deviance += np.std(data_block[:,k,4], axis=0) / np.mean(data_block[:,k,4], axis=0)
                if tot_deviance > min_std_green:
                    # We found constant cycle and varying green duration
                    if not detected:
                        # set new graph for plot:
                        fig = plt.figure(dpi=1200)
                        ax = plt.gca()  # plt.subplot(1, 1, 1)
                        plt.xlabel("Time in a fixed cycle", fontsize=Font_Size)
                        plt.ylabel("Number of Cycle", fontsize=Font_Size)
                        detected = True
                        # plot only once, the first time detected:
                        start_of_cycles = [green_data[start_cycle_indx[-prev_steps]-1+x*nmoves,1] for x in range(prev_steps-1)]
                        is_break = False
                        for k in range(nmoves):
                            if not isint(data_block[1,k,2]): # consider only non-pedestrian moves
                                continue
                            range1 = np.array([(data_block[x,k,3] - start_of_cycles[x]).seconds for x in range(prev_steps-1)])
                            if any([(data_block[x,k,3] < start_of_cycles[x]) for x in range(prev_steps-1)]):
                                is_break = True
                                break # it's starts before start of the cycle
                            range2 = np.minimum(np.array([(data_block[x,k,1] - start_of_cycles[x]).seconds for x in range(prev_steps-1)]), range1 + data_block[:,k,4]) #range1 + data_block[:,k,4]    #*datetime.timedelta(seconds=1)
                            plt.hlines(np.linspace(1,prev_steps-1, prev_steps-1) + k / max(20,nmoves+1), range1, range2, linewidth=1.5, colors=colors[k%15])
                            range3 = np.array([max(data_block[x, k, 3] + data_block[x, k, 4] - data_block[x, k, 1],0) for x in range(prev_steps - 1)])
                            plt.hlines(np.linspace(2, prev_steps, prev_steps-1) + k / max(20, nmoves + 1), np.zeros(prev_steps-1), range3, linewidth=1.5, colors=colors[k % 15])
                        curr_time = green_data[start_cycle_indx[-prev_steps],1].hour + green_data[start_cycle_indx[-prev_steps],1].minute / 60.0
                        fname = out_file + '_' + str(green_data[i-1, 0]) + '_' + str(curr_time) + '.png'
                        #if not os.path.exists(fname):
                        if not is_break:
                            plt.minorticks_on()
                            plt.xticks(fontsize=Font_Size)
                            mean_cyc = int(np.mean(cyc_num, axis=0))
                            plt.xlim([0, mean_cyc])
                            ax.set_xticks(np.append(ax.get_xticks(),mean_cyc)) # add tick for the cycle time.
                            plt.title('Zomet='+str(green_data[i-1, 0])+", Time="+green_data[start_cycle_indx[-prev_steps],1].strftime('%a %d/%m/%Y %H:%M:%S') +
                                      ', indxs:' + str(start_cycle_indx[-prev_steps]) + '÷' + str(i-1) + ', cyc='+str(mean_cyc), fontsize = 8)
                            plt.yticks(fontsize=Font_Size)
                            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                            # Show the major grid lines with dark grey lines
                            plt.grid(axis='x', b=True, which='major', color='#666666', linewidth=0.2, linestyle='-')
                            # Show the minor grid lines with very faint and almost transparent grey lines
                            plt.grid(axis='x', b=True, which='minor', color='#666666', linewidth=0.1, linestyle='-', alpha=0.9)
                            plt.savefig(fname)
                        plt.clf(); plt.cla(); plt.close(fig); plt.close('all'); gc.collect()  # clear previous content
                else:
                    detected = False



    print('Finished PeriodicityGreens3!')


def AnalyzeGreens(green_data, out_file, days, time_start, time_end, start_date, end_date):
    # Data: green_data[data points][0=#intersection, 1=cyc time stamp, 2=#move, 3=start green time, 4=green duration, 5=#prog].
    fig = plt.figure()
    cyc_durations = []
    time_anomality = []
    end_intergreens = []
    start_intergreens = []
    inner_intergreens = []
    switch_plans = 0
    cur_plan = green_data[0][5]
    start_cycle_indx = 0
    for i in range(1,green_data.shape[0]):
        if (green_data[i][3].weekday() in days):  # look only on specific days in the week
            diff_cycle = green_data[i][1].hour * 3600 + green_data[i][1].minute * 60.0 + green_data[i][1].second - \
                         (green_data[i - 1][1].hour * 3600 + green_data[i - 1][1].minute * 60.0 + green_data[i - 1][
                             1].second)
            if diff_cycle > 0:
                cyc_durations.append(diff_cycle)
                # store intergreens between start cycle and end of it:
                if (i-start_cycle_indx<20) and (start_cycle_indx>1): # reasonable number of datapoints in one cycle (bigger means disconnected data)
                   first_green = min(green_data[start_cycle_indx:i,3])
                   begin_diff = (first_green - green_data[start_cycle_indx-1,1]).seconds
                   #begin_diff = begin_diff.hour * 3600 + begin_diff.min * 60 + begin_diff.second
                   start_intergreens.append(begin_diff)
                   last_green = max(green_data[start_cycle_indx:i,3] + green_data[start_cycle_indx:i,4]*datetime.timedelta(seconds=1))
                   end_diff = (green_data[i-1,1]-last_green).seconds
                   #end_diff = end_diff.hour*3600 + end_diff.min*60 + end_diff.second
                   if green_data[i-1,1]<=last_green:
                       end_intergreens.append(0)
                   else:
                       end_intergreens.append(end_diff)
                   max_inner_intergreen = 0
                   for k in range(start_cycle_indx,i):
                       diff_list = green_data[k,3]-green_data[start_cycle_indx:i,3]-datetime.timedelta(seconds=1)*green_data[start_cycle_indx:i,4]
                       diff_list = np.append(diff_list, datetime.timedelta(seconds=10000)) # not to have empty list..
                       t = min([n for n in diff_list if n.days>=0]).seconds
                       if (t>max_inner_intergreen) and (t<10000):
                           max_inner_intergreen = t
                   inner_intergreens.append(max_inner_intergreen)
                start_cycle_indx = i # update new cycle to be this index
                if (diff_cycle < 40) or (diff_cycle > 150):
                    time_anomality.append(green_data[i,1].hour + green_data[i,1].minute / 60.0)
            if not (cur_plan == green_data[i,5]):
                cur_plan = green_data[i,5]
                switch_plans += 1
    # Creating plots
    plt.boxplot(cyc_durations, showfliers = False)
    plt.title("Cycle duration distribution")
    plt.show()
    fig = plt.figure()
    plt.boxplot(time_anomality, showfliers = False)
    plt.title("Time in a day of <72&>120sec cycles distribution")
    plt.show()
    fig = plt.figure()
    plt.boxplot(start_intergreens, showfliers = False)
    plt.title("Starting inter-green distribution")
    plt.show()
    fig = plt.figure()
    plt.boxplot(end_intergreens, showfliers = False)
    plt.title("Ending inter-green distribution")
    plt.show()
    fig = plt.figure()
    plt.boxplot(inner_intergreens, showfliers = False)
    plt.title("Maxmimum inner inter-green distribution")
    plt.show()
    print("Number of data points:", green_data.shape[0])
    print("Number of cycles:", len(cyc_durations))
    print("Number of switch between plans:", switch_plans)


def PeriodicityGreens4(green_data, out_file, days, time_start, time_end, start_date, end_date, junctions):
    # Save an array of data where previous fix #steps have constant cycle time but varying green durations
    # Data: green_data[data points][0=#intersection, 1=cyc time stamp, 2=#move, 3=start green time, 4=green duration, 5=#prog].
    curr_intersection = green_data[0][0]
    # curr_date = green_data[0][3].date()
    range_indxs = []  # store all found good ranges of data
    cmap = plt.get_cmap('tab20')     # see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    len_data = green_data.shape[0]
    # prepare for plot:
    Font_Size = 12
    fig = plt.figure(dpi=1200)
    last_start_cycle = 0
    start_arr = {}
    finish_arr = {}
    curr_date = 0

    for i in range(1, len_data):  # (1, len_data)
        # look only on the relevant data:
        if not (green_data[i, 0] in junctions):
            continue
        # if not (curr_date == green_data[i, 3].date()):
        #     print("changed date!")
        curr_date = green_data[i, 3].date()
        if not ((green_data[i, 3].weekday() in days) and (curr_date >= start_date) and (curr_date <= end_date) and isint(green_data[i, 2])):
            continue
            #curr_intersection = green_data[0, 0]
        curr_time = green_data[i, 3].hour + green_data[i, 3].minute / 60.0 + green_data[i, 3].second / 3600.0
        if not ((curr_time >= time_start) and (curr_time <= time_end)):
            continue
        diff_cycle = (green_data[i,1] - green_data[i - 1,1]).seconds
        if diff_cycle > 0:  # new cycle
            nmoves = i - last_start_cycle  # number of moves in the last cycle
            if any([(green_data[last_start_cycle+x,3] < green_data[last_start_cycle-1,1]) for x in range(nmoves)]) or (diff_cycle<60) or (diff_cycle>130):
                last_start_cycle = i
                continue
            curr_key = str(curr_date),str(green_data[i, 0])  # unique key for day and junction
            if not (curr_key in start_arr.keys()):
                start_arr[curr_key] = []
                finish_arr[curr_key] = []
            start_arr[curr_key].append(green_data[i-1, 1].hour + green_data[i-1, 1].minute / 60.0 + green_data[i-1, 1].second / 3600.0)
            finish_arr[curr_key].append(green_data[i, 1].hour + green_data[i, 1].minute / 60.0 + green_data[i, 1].second / 3600.0)
            last_start_cycle = i

    plt.clf()  # clear previous content
    plt.cla()
    plt.close(fig)
    plt.close('all')
    gc.collect()
    fig = plt.figure(dpi=1200)
    ax = plt.gca()  # plt.subplot(1, 1, 1)
    plt.title('Zmts=' + str(junctions) + ", Dates=" + start_date.strftime('%d/%m/%Y') +
              '÷' + end_date.strftime('%d/%m/%Y'), fontsize=8)
    plt.xlabel("Time in a 24hr day", fontsize=Font_Size)
    plt.ylabel("Day", fontsize=Font_Size)
    _,curr_inter = list(start_arr.keys())[0]
    x = 0
    y = 0
    for cur_date,cur_junc in start_arr.keys():  # seperated keys, by days
        if cur_junc==curr_inter:
            x += 1
        else: # new intersection
            x = 1
            y += 1/(len(junctions)+1)
            curr_inter = cur_junc
        plt.hlines(np.full(len(start_arr[cur_date, cur_junc][::2]), x + y), start_arr[cur_date, cur_junc][::2],
                   finish_arr[cur_date, cur_junc][::2], linewidth=2.5,
                   colors=colors[int(cur_junc) % 7])  # all even indexes 0,2,4,...
        plt.hlines(np.full(len(start_arr[cur_date, cur_junc][1::2]), x+y), start_arr[cur_date, cur_junc][1::2], finish_arr[cur_date, cur_junc][1::2], linewidth=2.5,
                   colors=colors[(int(cur_junc) % 7) + 7])  # all odd indexes 1,3,5,...
    plt.minorticks_on()
    plt.xticks(fontsize=Font_Size)
    plt.yticks(fontsize=Font_Size)
    ax.xaxis.set_major_locator(ticker.MultipleLocator((time_end - time_start) / 5.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator((time_end - time_start) / 50.0))
    # Show the major grid lines with dark grey lines
    plt.grid(axis='x', b=True, which='major', color='#666666', linewidth=0.2, linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.grid(axis='x', b=True, which='minor', color='#666666', linewidth=0.1, linestyle='-', alpha=0.9)

    fname = out_file + '_' + str(time_start) + '_' + str(time_end) + '_' + str(start_date) + '_' + str(end_date) + '.png'
    # os.chdir(folder_path)
    if not os.path.exists(fname):
        plt.savefig(fname)


    print('Finished PeriodicityGreens4!')


def PeriodicityGreens5(in_file,out_file, sheets, rang, is_seperate, lw, time_start, time_end, start_date, end_date):
    from matplotlib.legend_handler import HandlerLine2D
    # About indexing and slicing input:
    # https://stackoverflow.com/questions/509211/understanding-slice-notation
    # Save an array of data where previous fix #steps have constant cycle time but varying green durations
    # Data: green_data[data points][0=#intersection, 1=cyc time stamp, 2=#move, 3=start green time, 4=green duration, 5=#prog].
    links_of_interest = [88, 241, 248, 49, 173, 245, 251, 16, 34, 249, 256, 27, 116, 216, 252, 28, 117, 176, 262, 6,
                         260, 266, 12, 187, 263, 153, 254, 258, 24, 239]
    # links_of_interest = [2, 5, 24, 51, 66, 92, 104, 107, 152, 156, 159, 160, 161, 162, 164, 166, 167, 168]
    links_of_interest = [x - 2 for x in links_of_interest]
    cmap = plt.get_cmap('tab20')     # see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    df = np.load(in_file, allow_pickle=True).item()
    len_data = len(df["StepsStart"])
    df1 = np.concatenate(df["StepsStart"])
    df1 = np.sort(df1)
    num_file = int(os.path.splitext(in_file)[0][-1])
    if num_file>=1:  # Data file 01 or above is already contain only links_of_interest
        links_of_interest = np.linspace(0,len_data-1,len_data, dtype=int)
    # prepare for plot:
    Font_Size = 12
    fig = plt.figure(dpi=1200)
    line_style = ['solid', 'dashed', 'dashdot', 'dotted']
    line_style2 = ['-', '--', '-.', ':']
    twin = ax2 = ax = 0
    start_arr, finish_arr, val_arr, link_arr, start_datetime = [], [], [], [], []

    for indx, i in enumerate(np.array(links_of_interest)[rang]):
        len_link = df["StepsStart"][i].shape[0]
        for j in range(len_link):   # go over link's array
            curr_start = datetime.datetime.fromtimestamp(df["StepsStart"][i][j])
            curr_finish = datetime.datetime.fromtimestamp(df["StepsEnd"][i][j])
            if (curr_start.date() >= start_date) and (curr_start.date() <= end_date) and (curr_start.hour >= time_start) and (curr_start.hour < time_end):
                start_datetime.append(curr_start)
                start_t = curr_start.hour + curr_start.minute / 60.0 + curr_start.second / 3600.0
                end_t = curr_finish.hour + curr_finish.minute / 60.0 + curr_finish.second / 3600.0
                start_arr.append(start_t)
                if curr_start.hour==23 and curr_finish.hour==0:
                    end_t += 24
                finish_arr.append(end_t)
                link_arr.append(indx)  # it's a must array, to seperate graphs by links
                comb = []
                for sheet in sheets:
                    if (sheet=="StepsStart"):
                        comb.append(indx+0.5)
                    else:
                        if (sheet=="Plans"):
                            comb.append(df[sheet][i][j] % 100)  # it affects if we use SP coding of link+original SP ID...
                        else:
                            comb.append(df[sheet][i][j])
                val_arr.append(np.stack(comb))

    # Now sort all arrays by time:
    start_datetime = np.array(start_datetime)
    start_arr = np.array(start_arr)
    finish_arr = np.array(finish_arr)
    val_arr = np.array(val_arr)
    link_arr = np.array(link_arr)
    # org_data = np.argsort(start_datetime, axis=None)  # indexes of the sorted array
    # start_arr = start_arr[org_data]
    # finish_arr = finish_arr[org_data]
    # val_arr = val_arr[org_data]
    # link_arr = link_arr[org_data]
    # start_datetime = start_datetime[org_data]
    start_link = np.stack([start_datetime,link_arr])

    # start_indx = 0
    # finish_indx = -1
    for i in range((end_date-start_date).days+1):  # go over each day
        print("Image File: "+str(i+1)+"/"+str((end_date-start_date).days+1))
        curr_date = start_date + i*datetime.timedelta(days=1)
        plt.clf()  # clear previous content
        plt.cla()
        plt.close(fig)
        plt.close('all')
        del twin, ax2, ax, fig
        gc.collect()
        time.sleep(1)
        gc.collect()
        #fig = plt.figure(dpi=1500)
        # Add subplots for different axes dimensions
        if is_seperate:
            tot_axes = len(np.array(links_of_interest)[rang])
            x_axes = int(sqrt(tot_axes))
            y_axes = int(tot_axes//x_axes)
            fig, ax2 = plt.subplots(x_axes, y_axes, dpi=1500, figsize=[10.0, 4.8])
            fig.subplots_adjust(left=0.03, right=0.98)
        else:
            fig, ax2 = plt.subplots(dpi=1500, figsize=[10.0, 4.8])  # figzise = window size (not the plot's size)
            ax2.set_ylabel(sheets[0])
            # It's how much the plot is in percentage from the width of the whole window (default: left=0, right=1)
            tot_axes = len(sheets) - 1
            fig.subplots_adjust(left=0.2, right=0.75)
        twin = []
        axes_pos = 1.0
        # Add more axes, each with its own scale: https://matplotlib.org/stable/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
        for x in range(tot_axes):
            if is_seperate:
                twin.append(ax2[x//y_axes,x%y_axes])
            else:
                twin.append(ax2.twinx())
                if x>0:
                    twin[-1].spines["right"].set_position(('axes', axes_pos))
                twin[-1].set_ylabel(sheets[x+1])
            axes_pos += 0.1
        if not is_seperate:
            twin.insert(0,ax2)

        ax = plt.gca()  # plt.subplot(1, 1, 1)
        tittle = "Date=" + curr_date.strftime('%d/%m/%Y %A') + ", Data-File=" + str(num_file)
        # if "Plans" in sheets:
        #     arra = np.concatenate(df["Plans"])
        #     arra = np.where(np.isnan(arra), -1, arra)
        #     found_seqs,freqs  = np.unique(arra, return_counts=True)
        #     tittle = tittle + "\n(Value,freq)=" + str(["("+str(found_seqs[i]) +","+ str(round(freqs[i]/sum(freqs)*100.0,2))+"%) " for i in range(len(found_seqs))])
        plt.title(tittle, fontsize=8)
        plt.xlabel("Time in a 24hr day", fontsize=Font_Size)
        #plt.ylabel("Link", fontsize=Font_Size)
        mm = 0
        for indx, j in enumerate(np.array(links_of_interest)[rang]):  # go over each link
            indexes = [x for x in range(start_link.shape[1]) if start_link[0,x].date()==curr_date and start_link[1,x]==indx]
            if indexes==[]:  # date is out of valid weekdays
                mm += 1
                continue

            for inx, sheet in enumerate(sheets):
                if is_seperate:
                    inxx = indx
                else:
                    inxx = inx
                if sheets == ["StepsStart"]: # swapping colors every point, for each link
                    plt.hlines(val_arr[indexes[::2],inx], start_arr[indexes[::2]], finish_arr[indexes[::2]],
                               linewidth=lw, colors=colors[indx % 7], linestyles=line_style[inx % 4],
                               label=sheet + "_link:" + str(j))  # all even indexes 0,2,4,...
                    plt.hlines(val_arr[indexes[1::2],inx], start_arr[indexes[1::2]], finish_arr[indexes[1::2]],
                               linewidth=lw, colors=colors[(indx % 7) + 7], linestyles=line_style[inx % 4],
                               label=sheet + "_link:" + str(j))  # all odd indexes 1,3,5,...
                else:
                    twin[inxx].errorbar((start_arr[indexes] + finish_arr[indexes]) / 2.0, val_arr[indexes, inx],
                                             xerr=(finish_arr[indexes] - start_arr[indexes]) / 2.0,
                                             linewidth=lw, color=colors[indx*len(sheets)+inx % 15], linestyle=line_style2[inx % 4], ms=lw*2,
                                             marker="o", label=sheet + "_link:" + str(j))
                    if is_seperate:
                        twin[inxx].legend(fontsize=Font_Size / 4, handlelength=12)
                    else:
                        twin[inxx].legend(fontsize=Font_Size/4, loc='upper center', bbox_to_anchor=(-0.25, 1.05-1.0/len(twin)*(inx+1)), handlelength=12)

        if mm == len(np.array(links_of_interest)[rang]):  # nothing to show
            twin = ax2 = ax = None
            continue
        plt.minorticks_on()
        plt.xticks(fontsize=Font_Size)
        plt.yticks(fontsize=Font_Size)
        if not is_seperate:
            ax2.minorticks_on()
            ax2.xaxis.set_major_locator(ticker.MultipleLocator((time_end - time_start) / 5.0))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator((time_end - time_start) / 50.0))
            ax2.grid(axis='x', b=True, which='major', color='#666666', linewidth=0.2, linestyle='-')
            ax2.grid(axis='x', b=True, which='minor', color='#666666', linewidth=0.1, linestyle='-', alpha=0.9)
        # Show the major grid lines with dark grey lines
        plt.grid(axis='x', b=True, which='major', color='#666666', linewidth=0.2, linestyle='-')
        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.grid(axis='x', b=True, which='minor', color='#666666', linewidth=0.1, linestyle='-', alpha=0.9)
        if is_seperate:
            plt.legend(fontsize=Font_Size / 4, handlelength=12)
        else:
            plt.legend(fontsize=Font_Size/4, loc='upper center', bbox_to_anchor=(-0.25, 1.05), handlelength=12)  # bbox=(relative to x-axis, relative to y-axis)

        fname = out_file + '_' + str(time_start) + '-' + str(time_end) + '_' + str(curr_date) + '_' + str(num_file) + '.png'
        # os.chdir(folder_path)
        #if not os.path.exists(fname):
        time.sleep(1)
        plt.savefig(fname)
        time.sleep(2)

    print('Finished PeriodicityGreens5!')



def PeriodicityGreens6(in_file,out_file, res_file, sheets, rang, lw, time_start, time_end, start_date, end_date):
    from matplotlib.legend_handler import HandlerLine2D
    # NO "StepsStart" allowed as input sheet
    # Compare prepared data with resulted data from training and evaluated by DL models
    # Data: green_data[data points][0=#intersection, 1=cyc time stamp, 2=#move, 3=start green time, 4=green duration, 5=#prog].
    cmap = plt.get_cmap('tab20')     # see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    df = np.load(in_file, allow_pickle=True).item()   # Data file, to get the original data
    len_data = len(df["StepsStart"])
    df_sheets = []
    org_data = np.argsort(np.concatenate(df["StepsStart"]), axis=None)
    for sheet in sheets+["StepsStart", "StepsEnd"]:  # last one is StepsStart&End
        if sheet in df:
            df_sheets.append(np.concatenate(df[sheet])[org_data])
    num_file = int(os.path.splitext(in_file)[0][-1])
    links_of_interest = np.linspace(0,len_data-1,len_data, dtype=int)
    temp = os.path.splitext(data_fname_output)
    df_info = np.load(temp[0] + "_06" + temp[1], allow_pickle=True).item()  # Data set prepared file
    output_indexes = df_info["output_indexes"]
    out_sheets = df_info["out_sheets"]
    # prepare for plot:
    Font_Size = 12
    fig = plt.figure(dpi=1200)
    line_style = ['solid', 'dashed', 'dashdot', 'dotted']
    line_style2 = ['-', '--', '-.', ':']
    twin = ax2 = ax = 0
    start_arr, finish_arr, val_arr, link_arr, start_datetime = [], [], [], [], []
    res_data = torch.load(res_file)
    ya = res_data["ya"]              # a list of y actual data blocks
    y_preda = res_data["y_preda"]    # a list of y predicted data blocks
    # TO DO: find all 3 datas y, y_pred, and original in the same ranges of dates and times in the input, and the same sheets... df_info["sheets"] and indexes...

    # x,y dim = [#samples, input_channels,   #time steps, #nodes]
    for i in range(ya.shape[0]):   # go over all y's
        #len_link = df["StepsStart"][i].shape[0]
        for j in range(ya.shape[3]):   # go over all y's links
            curr_index = int(ya[i,-1,0,j])
            curr_start = datetime.datetime.fromtimestamp(df_sheets[-2][curr_index])  # StepsStart (assuming 1=time step)
            curr_finish = datetime.datetime.fromtimestamp(df_sheets[-1][curr_index])  # StepsEnd (assuming 1=time step)
            if (curr_start.date() >= start_date) and (curr_start.date() <= end_date) and (curr_start.hour >= time_start) and (curr_start.hour < time_end):
                start_datetime.append(curr_start)
                start_t = curr_start.hour + curr_start.minute / 60.0 + curr_start.second / 3600.0
                end_t = curr_finish.hour + curr_finish.minute / 60.0 + curr_finish.second / 3600.0
                start_arr.append(start_t)
                if curr_start.hour==23 and curr_finish.hour==0:
                    end_t += 24
                finish_arr.append(end_t)
                link_arr.append(j)  # it's a must array, to seperate graphs by links
                comb = []
                for indx, sheet in enumerate(sheets):
                    # first add original data (if exists)
                    if (sheet=="Plans"):
                        comb.append(df_sheets[indx][curr_index] % 100)  # it affects if we use SP coding of link+original SP ID...
                    elif sheet in df:
                        pass # comb.append(df_sheets[indx][curr_index])  # if it's identical as ACTUAL then it's redundant
                    # then add actual and predicted values from the model
                    if sheet in out_sheets:
                        curr_index2 = out_sheets.index(sheet)
                        comb.append(ya[i,curr_index2,0,j])
                        comb.append(y_preda[i,curr_index2,0,j])
                val_arr.append(np.stack(comb))

    new_sheets = []
    for sheet in sheets:
        # first add original data (if exists)
        if sheet in df:
            pass # new_sheets.append(sheet)  # original data
        # then add actual and predicted values from the model
        if sheet in out_sheets:
            new_sheets.append("T:"+sheet)  # ground-truth (suppose to be = original data)
            new_sheets.append("P:"+sheet)  # predicted
    # Now sort all arrays by time:
    start_datetime = np.array(start_datetime)
    start_arr = np.array(start_arr)
    finish_arr = np.array(finish_arr)
    val_arr = np.array(val_arr)
    link_arr = np.array(link_arr)
    # org_data = np.argsort(start_datetime, axis=None)  # indexes of the sorted array
    # start_arr = start_arr[org_data]
    # finish_arr = finish_arr[org_data]
    # val_arr = val_arr[org_data]
    # link_arr = link_arr[org_data]
    # start_datetime = start_datetime[org_data]
    start_link = np.stack([start_datetime,link_arr])

    # start_indx = 0
    # finish_indx = -1
    for i in range((end_date-start_date).days+1):  # go over each day
        print("Image File: "+str(i+1)+"/"+str((end_date-start_date).days+1))
        curr_date = start_date + i*datetime.timedelta(days=1)
        plt.clf()  # clear previous content
        plt.cla()
        plt.close(fig)
        plt.close('all')
        del ax, fig
        gc.collect();    time.sleep(1);    gc.collect()
        fig = plt.figure(dpi=1200)
        ax = plt.gca()  # plt.subplot(1, 1, 1)
        tittle = "Date=" + curr_date.strftime('%d/%m/%Y %A') + ", Data-File=" + str(num_file)
        # if "Plans" in sheets:
        #     arra = np.concatenate(df["Plans"])
        #     arra = np.where(np.isnan(arra), -1, arra)
        #     found_seqs,freqs  = np.unique(arra, return_counts=True)
        #     tittle = tittle + "\n(Value,freq)=" + str(["("+str(found_seqs[i]) +","+ str(round(freqs[i]/sum(freqs)*100.0,2))+"%) " for i in range(len(found_seqs))])
        plt.title(tittle, fontsize=8)
        plt.xlabel("Time in a 24hr day", fontsize=Font_Size)
        #plt.ylabel("Link", fontsize=Font_Size)
        mm = 0
        for indx, j in enumerate(np.array(links_of_interest)[rang]):  # go over each link
            indexes = np.array([x for x in range(start_link.shape[1]) if start_link[0,x].date()==curr_date and start_link[1,x]==indx])
            if indexes.size==0:  # date is out of valid weekdays
                mm += 1
                continue
            org_data = np.argsort(start_arr[indexes], axis=None)
            indexes = indexes[org_data]
            for inx, sheet in enumerate(new_sheets):
                ax.errorbar((start_arr[indexes] + finish_arr[indexes]) / 2.0, val_arr[indexes, inx],
                                             xerr=(finish_arr[indexes] - start_arr[indexes]) / 2.0,
                                             linewidth=lw, color=colors[indx % 15], linestyle=line_style2[inx % 4], ms=lw*2,
                                             marker="o", label=sheet + "_link:" + str(j))
                ax.legend(fontsize=Font_Size/4, loc='upper center', bbox_to_anchor=(-0.25, 1.05-1.0/3*(inx+1)), handlelength=12)

        if mm == len(np.array(links_of_interest)[rang]):  # nothing to show
            twin = ax2 = ax = None
            continue
        plt.minorticks_on()
        plt.xticks(fontsize=Font_Size)
        plt.yticks(fontsize=Font_Size)
        ax.minorticks_on()
        ax.xaxis.set_major_locator(ticker.MultipleLocator((time_end - time_start) / 5.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator((time_end - time_start) / 50.0))
        ax.grid(axis='x', b=True, which='major', color='#666666', linewidth=0.2, linestyle='-')
        ax.grid(axis='x', b=True, which='minor', color='#666666', linewidth=0.1, linestyle='-', alpha=0.9)
        # Show the major grid lines with dark grey lines
        plt.grid(axis='x', b=True, which='major', color='#666666', linewidth=0.2, linestyle='-')
        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.grid(axis='x', b=True, which='minor', color='#666666', linewidth=0.1, linestyle='-', alpha=0.9)
        plt.legend(fontsize=Font_Size/4, loc='upper center', bbox_to_anchor=(-0.25, 1.05), handlelength=12)  # bbox=(relative to x-axis, relative to y-axis)

        fname = out_file + '_' + str(time_start) + '-' + str(time_end) + '_' + str(curr_date) + '_' + str(num_file) + '.png'
        # os.chdir(folder_path)
        #if not os.path.exists(fname):
        time.sleep(1)
        plt.savefig(fname)
        time.sleep(2)

    print('Finished PeriodicityGreens6!')


def PrintFreqOfIntersections(folder_path):
    # Accumulate the #times each source->target intersections appear in BT data files, in some folder, and store in some file.

    regex = re.compile(".*\.csv")
    #folder_content = os.listdir(folder_path) # Create the list of all your folder content with os.listdir()
    filenames = [os.path.join(root,file) for root, dirs, files in os.walk(folder_path) for file in files if regex.search(file)]  #
    if os.path.exists("Links.npy"):
        df = np.load("Links.npy", allow_pickle=True).item()
        filenames = [x for x in filenames if x not in df["Info"]]
    else:
        df = {}
        # 8 Columns: link's name; length; freq; from; to; moves1; moves2; moves3
        df["Links"] = [[] for _ in range(8)]
        df["Info"] = [] # include all files of BT data and current time

    for indx, fname in enumerate(filenames):
        BT = ConvertCsvtoNpy(fname, False)
        print("Start File: " + fname + ", " + str(indx+1) + "/" + str(len(filenames)))
        len_data = BT.shape[0]
        add_col = 0
        if (BT[0, 0] == BT[0, 2]) or (len(BT[0])==22):  # sometimes there's additional redundant column.
            add_col = 1
        # tot_valid = 0    # Different analysis needed here, since it's limited to SP data points

        for i in range(1, len_data):  # (1, len_data)
            if (np.isnan(BT[i, 4 + add_col])) or (np.isnan(BT[i, 5 + add_col])) or (not (BT[i, 19 + add_col] == "Valid")):
                continue  # if the value of entering/exiting node is NaN or if trip status isn't Valid then skip this row
            if not((BT[i, 4 + add_col][0:2]=="TA") and (BT[i, 5 + add_col][0:2]=="TA")):
                continue  # must be both intersections starting with "TA" initials
            from_intersection = BT[i, 4 + add_col][2:]  # Remove "TA" initials in the string
            to_intersection = BT[i, 5 + add_col][2:]    # Remove "TA" initials in the string
            Link_Name = BT[i, 4 + add_col] + BT[i, 5 + add_col]
            curr_link = np.squeeze(np.where(np.asarray(df["Links"][0]) == Link_Name))  # search the indx of the link in nodes file, given enter and exit nodes.
            if (curr_link.size > 0):  # if such link exists in the table
                df["Links"][2][curr_link] += 1    # increase freq;
            else: # if no such link, then we add it
                df["Links"][0].append(Link_Name)  # link's name;
                df["Links"][1].append(0)          # length;
                df["Links"][2].append(0)          # freq;
                df["Links"][3].append(from_intersection)  # from;
                df["Links"][4].append(to_intersection)    # to;
                df["Links"][5].append(0)          # moves1;
                df["Links"][6].append(0)          # moves2;
                df["Links"][7].append(0)          # moves3

        df["Info"].append(fname)
        np.save("Links.npy", df)
        del BT; gc.collect()


    print('Finished PrintFreqOfIntersections!')


##-----------------------------------------------------------------------------------------------------------------------##
# Introduction:
# Files have one “Info” tab to gather all different parameters of SQL and BT data. And we first create “nan” data file,
# of all the 5-min averaged data from BT and SQL. Then we apply post processing for the final data file (without the “nan” suffix),
# which includes smoothing by linear interpolation and/or averaging sliding window.
#
# In all procedures (BT data extraction, SQL data extraction and post-processing) we either create new sheets in our excel data file,
# or update the existing one, or leave the other sheets in place (which are not updated).
#
# “nan”:
# In BT data extracting procedure “TransformBT_graph3“, we first put the data in small arrays and only after that we put it in
# the big matrix of the year period, since we have to apply division operation of pairs of arrays (though theoretically we can do it
# for the specific range in the big matrix, but we left it as it is).
# In SQL data extracting procedure “TransformBT_graph4”, we also create full-year green and plans tables, or load them from
# existing tabs in the data excel file. Each row we want to update we check it first: if there’s nan we initialize it to 0, so that
# we can accumulate green time (over the 5-min period), if it’s not nan then we erase it to be 0 for further accumulation.
# We do so assuming that there is no mixing between sql files, i.e. we could not update previous data rows. So it must be previous
# wrong runnings which we wish to remove and replace by the new data.
# After gathering all plan data for every 5-min period, in the format e.g. “7,7,7,7” which express all the traffic plans that were
# applied in this 5-min period, we perform maximum count operation over each such cell. I.e. we take the most frequent plan to represent
# every cell. E.g. “5,7,7,5,3,5” ends up to be 5.








##-----------------------------------------------------------------------------------------------------------------------##


#------- Define all files and their folders:
python_path = r"D:\Doctorate\Google Drive\My work\My Python\STGCN"
links_name = "Linksb.npy" # Adjecancy matrix, file containing all roads (nodes in graph) exists in data file
matrix1_name = "adj03_nodes_adj.csv"
matrix2_name = "adj03_links_adj.csv" #"STGCN-PyTorch-master2/dataset/W_228.csv" #"adj01_links_adj.csv"  # Adjecancy matrix
matrix2_name_npy = "adj03_links_adj.npy"
# input file:
sql_input_path = r"C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA" # r"D:\Doctorate\Python\BTDATA\BT-Data"
sql_input_name = "CommonDB.mdf" #"STGCN-PyTorch-master2/dataset/V_228.csv" #"MAY_23_2020_to_PRESENT_seq300_02.csv" #"Data part1.csv"
csv_input_path = r"D:\Doctorate\Python\BTDATA\BT-Data\NEW DATA\2019\JuneToNov2019" # r"D:\Doctorate\Python\BTDATA\BT-Data"
code_input_name = "Cross_Info.npy"
data_path = r"D:\Doctorate\Python\BTDATA\BT-Data"
data_name = "Data_2019.npy" #"STGCN-PyTorch-master2/dataset/V_228.csv" #"MAY_23_2020_to_PRESENT_seq300_02.csv" #"Data part1.csv"
# final full file names:
sql_fname_input = os.path.join(sql_input_path,sql_input_name)
code_fname_input = os.path.join(data_path,code_input_name)
#csv_fname_input2 = os.path.join(csv_input_path,csv_input_name2)
data_fname_output = os.path.join(data_path,data_name)  # file with our data
links_fname = os.path.join(python_path,links_name) # file containing all roads (nodes in graph) exists in data file
matrix1_fname = os.path.join(python_path,matrix1_name)  # Adjacency matrix of nodes
matrix2_fname = os.path.join(python_path,matrix2_name)  # Adjacency matrix of links
matrix2_fname_npy = os.path.join(python_path,matrix2_name_npy)
SP_Files = os.path.join(data_path,'SP_DATA')   # replace the huge SQL file with several NPY files

#------- Define for sheets/features:
input_sheets = ["GreenTimes", "Velocity", "Time_month", "Time_day", "Time_weekday", "Time_hour"]  # "Velocity", "Plans", "GreenTimes", "nVehs", "Plans_coded"
output_sheets = ["GreenTimes"]
# for "Post_process_smooth" (Create Data5 file):
smoothings = [(True, True),(True, True)]  # only for SP&BT input-and-then-output features
avg_neighbors = 3   # number of neighbors to average for smoothing the data, must be odd number: 1,3,5,..
max_NaNs = 6        # maximum allowed series of NaNs for interpolation
# for "Post_DataSets", include also I/O_sheets above (Create Data6 and data-set files):
n_train, n_val, n_test = 70, 15, 15   # percentages for splitting data points into 3 data sets
perm = 2333  # -1=leave time order for the 3 data sets, otherwise=random order (seed number)
SP_coding = 1  # 0: coding into groups based on description, 1: coding based on link+SP ID
max_std_cyc = 0.2        # maximum mean/std deviation in cycle durations (in search for data sequences)
max_dev_threshold = 20   # maximum deviation between start of cycles (in search for data sequences)
pred_steps = 0           # prediction step relative to the current step (0,1,...)
seq_length = 13          # total sequence length (previous+prediction steps)
n_his = seq_length - pred_steps - 1
# for "Post_Data_SP" (Create Data3 file):
max_threshold = 20                    # maximum allowable remainder for last fake cycle to combine with real cycle (in search to complete missing cycles)
max_threshold2 = max_threshold/2.0    # maximum allowable cycle durations deviation (in search to complete missing cycles)
max_cycles = 6                        # maximum allowable cycles in the gap between 2 valid cycles (in search to complete missing cycles)
sheet_code, sheet_group, is_complete = 'tblPrograms', 'tblPrograms_group', True  # for encoding "Plans" feature

continuous_sheets = ["GreenTimes","Velocity","nVehs","TravelTime","Density","Flow"]
BT_sheets = ["Velocity", "nVehs", "TravelTime", "Density", "Flow"]
SP_sheets = ["GreenTimes", "Plans", "StepsStart", "StepsEnd"]
weekdays = [0,1,2,3,6]   # allowable weekdays in the data Monday=0, Sunday=6
start = datetime.date(2019, 6, 1)     # starting allowable date in data
finish = datetime.date(2019, 12, 1)   # finishing allowable date in data
average_for_seconds = 300  # number of seconds the data is averaged to



# stuff to run always here such as class/def
def main():
    # f = open(os.path.join(python_path,"log"+datetime.datetime.now().strftime("%d.%m.%Y_%H_%M_%S")+".txt"), 'w')
    # sys.stdout = Tee(sys.stdout, f)
    # Stages for cycle-based data pre-processing: (DELETE PREVIOUS DATA FILES)------------------------------------------------------
    # now = datetime.datetime.now(); current_time = now.strftime("%H:%M:%S"); print("Current Time =", current_time)
    # TransformBT_Files() # Create Data1, and NPY files from BT files
    # now = datetime.datetime.now(); current_time = now.strftime("%H:%M:%S"); print("Current Time =", current_time)
    # TransformSP()  # Create Data2
    # now = datetime.datetime.now(); current_time = now.strftime("%H:%M:%S"); print("Current Time =", current_time)
    # Post_Data_SP()  # Create Data3
    # now = datetime.datetime.now(); current_time = now.strftime("%H:%M:%S"); print("Current Time =", current_time)
    # TransformBT()  # Create Data4 (must delete Data4 if restarting, since it store after every CSV long processing..)
    # now = datetime.datetime.now(); current_time = now.strftime("%H:%M:%S"); print("Current Time =", current_time)
    # Post_process_smooth()  #  Create Data5
    # now = datetime.datetime.now(); current_time = now.strftime("%H:%M:%S"); print("Current Time =", current_time)
    # Post_DataSets()  # Create Data6 and 3 data set files
    # now = datetime.datetime.now(); current_time = now.strftime("%H:%M:%S"); print("Current Time =", current_time)
    # Permute_DataSets()

    def excel_time(ttime):
        # convert ttime=rows in excel file of data, into its actual time:
        dt = datetime.datetime.fromtimestamp(1561938900+ttime*5*60)
        print(dt)
    def real_time(dt):
        # convert actual ttime (e.g. '2019-07-08 18:20:00') to excel time=rows in excel file of data:
        ttime = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        ttime = datetime.datetime.timestamp(ttime)
        ttime = int((ttime - 1561938900)/300)
        print(ttime)

    PeriodicityGreens5(os.path.join(data_path, "Data_2019_05.npy"), os.path.join(csv_input_path,'Per5'),["Velocity"],#["Flow", "nVehs", "StepsStart", "GreenTimes","Velocity", "Plans", "TravelTime],
                       [1,5,10,16],True, 1,0, 24, datetime.date(2019, 7, 1), datetime.date(2019, 12, 1))  # slice(None)  [1,5,10,16] [5,14,20,27,29] [1,5,10,14,18,20,24,27,29]  [3,4,12,13,9]

    # PeriodicityGreens6(os.path.join(data_path, "Data_2019_05.npy"), os.path.join(csv_input_path, 'Per6'), os.path.join(python_path, 'model_STGCN4.pt'),
    #                    ["GreenTimes"],
    #                    [1,5,10,14,18,20,24,27,29],2,0, 24, datetime.date(2019, 7, 1), datetime.date(2019, 12, 1))  # [1,5,10,14,18,20,24,27,29]
    # PrintFreqOfIntersections(r"D:\Doctorate\Python\BTDATA\BT-Data\NEW DATA")
    # ConvertNpytoExcel("Links.npy")
    # ConvertExceltoNpy("Linksb.xlsx", True)

    print("Finished!")
    # f.close()


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()