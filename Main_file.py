import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, ChebConv, GATConv
import torch_geometric.transforms as T
#import re # for wildcard search
import gc

name=""

# Loading data functions:  ------------------------------------------------------------------------------------------

def TotalRows(file_path):
    df = pd.read_excel(file_path, sheet_name=op_sheets[0], header=None).values.astype(float)
    return len(df)

def ExtractSheets(file_path):
    # Extract all sheets that are not starting with "Info_"
    xl = pd.ExcelFile(file_path)
    op_sheets = []
    for curr_sheet in xl.sheet_names:
        #if not(curr_sheet[:5]=="Info_"):
        if curr_sheet in input_sheets:
            op_sheets.append(curr_sheet)
        if curr_sheet in output_sheets:
            op_sheets.append(curr_sheet)
    return op_sheets

def ExtractMultiData(file_path, max_sheets):
    global input_sheets
    global input_diff
    # extract data from 2d arrays for each sheet, into one 3d numpy array
    '''
    df = []
    number_sheet = 0
    for curr_sheet in op_sheets:
        if number_sheet<max_sheets:
            df.append(pd.read_excel(file_path, sheet_name=curr_sheet, header=None))#.astype(float))  since there also int types
            if curr_sheet == 'Plans_coded':
                df[-1] = df[-1].fillna(0)
        number_sheet += 1
    ranges = Find_CrossRanges2(df, 0, 0.7, 3, 2000)
    '''
    dff = np.load(file_path, allow_pickle=True).item()
    if 'Plans_coded' in list(dff.keys()):
        dff['Plans_coded'][dff['Plans_coded'] == np.nan] = 0
    df = []
    plans_indx = 0
    for curr_sheet in op_sheets:
        df.append(dff[curr_sheet])
        if curr_sheet == 'Plans_coded':
            plans_indx = len(df)-1
    #df = {key: df[key] for key in list(df.keys()) & op_sheets}

    ranges = Find_CrossRanges2(df, 0, 0.7, 3, 2000)

    if (is_binary) and ('Plans_coded' in list(dff.keys())):
        # If we want to split the binary channel into its bits channels:
        input_diff = [x for x in range(len(input_sheets))]
        width = 6  # bit width, or the number of groups
        llx = df[plans_indx].shape[0]
        lly = df[plans_indx].shape[1]
        new_channels = np.zeros((llx, lly, width), dtype=np.int32)
        #for ix in range(llx):
        #    for iy in range(lly):
        for ix,iy in np.ndindex(df[plans_indx].shape):
            new_channels[ix,iy,:] = [int(x) for x in '{:0{size}b}'.format(df[plans_indx][ix,iy], size=width)]
        df.pop(plans_indx)
        input_sheets.pop(plans_indx)
        diff_code = input_diff[plans_indx]
        input_diff.pop(plans_indx)
        for ii in range(width):
            df.insert(plans_indx, new_channels[:,:,ii])
            input_sheets.append('Plans_coded'+str(ii))
            input_diff.append(diff_code)
    return ranges, np.dstack(df)

def load_matrix(file_path):
    return np.load(file_path, allow_pickle=True) #pd.read_csv(file_path, header=None).values.astype(float)

def Adjacency_Matrix(Win, trans):
    global L
    global Lk
    global edge_index
    global edge_weight
    # transform unweighted adjacency matrix into either weighted by distances [1,0,0], or by dijastra shortest path between each pair of nodes [0,1,0], or by exponential [0,0,1], or combination of those
    Wout = np.zeros(Win.shape)
    #trans = [0, 1, 1, 1]
    if trans[0]==1:  # return symmetric Wout
        for xx, yy in np.ndindex(Win.shape):
            Wout[xx, yy] = int(Win[xx,yy]) | int(Win[yy,xx])
    if trans[1]==1:  # return Wout of distances between all connected pairs
        A = Win.copy()
        if not (np.all(Wout == 0)):
            A = Wout.copy()
        node_data = np.load(links_fname, allow_pickle=True).item()
        node_data = node_data[list(node_data.keys())[0]]
        nodes_length = node_data.shape[0] # len(node_data[0])
        xx, yy = np.where(A == 1)
        for ii in range(len(xx)):
            Wout[xx[ii],yy[ii]] = (node_data[xx[ii],1] + node_data[yy[ii],1])/2
    if trans[2]==1: # return Wout of shortest distances between all nodes
        import networkx as nx
        A = Win.copy()
        if not(np.all(Wout==0)):
            A = Wout.copy()
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        for xx,yy in np.ndindex(A.shape): #xx,yy in enumerate(A):
            try:
                Wout[xx,yy] = nx.dijkstra_path_length(G,xx,yy)
            except nx.NetworkXNoPath:
                Wout[xx,yy] = 1e9  # for those that have no path, represents infinity..
    if trans[3]==1:  # return exponential weighted adjacency matrix, see https://arxiv.org/pdf/1709.04875.pdf
        A = Win.copy()
        if not(np.all(Wout==0)):
            A = Wout.copy()
            Wout[:,:] = 0
        for xx,yy in np.ndindex(A.shape): #xx,yy in enumerate(A):
            res = np.exp(-np.power(A[xx,yy],2)/sqr_sigma)
            if not(xx==yy) and res>=epsilon:
                Wout[xx,yy] = res
        plt.imshow(Wout, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title = 'Adjacency Matrix Hot Map'
        plt.show()
        plt.savefig('AdjMatrix.jpg')
    # Graph
    L = scaled_laplacian(Wout)
    Lk = cheb_poly(L, Ks)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
    # convert Adj. matrix to COO form:
    edge_index = torch.from_numpy(Wout).to_sparse()
    # edge_index = edge_index.numpy()
    edge_index = edge_index.indices().to(device)  # without values
    edge_weight = None

    return Wout


def Find_CrossRanges2(df, init_indx, min_perc, max_range, min_range):
    # search for continuous range of not nan rows, of at least min_range rows, starting searching from init_indx row
    aggregate = False
    starting_range = 0
    finish_range = 100000
    range_width = 0
    ranges = []
    ndd = []
    ldf = range(len(df))
    for ii in ldf:
        ndd.append(sum(np.isnan(np.transpose(df[ii]))))  # number of nans at each row
    threshold = int((1 - min_perc) * df[0].shape[1])  # max number of nans allowed
    rr = df[0].shape[0]
    for indx in range(rr):
        if indx>=init_indx:
            #if indx>70000:
            #    indx = indx
            all_nan = True
            for ii in ldf:
                all_nan = all_nan and (ndd[ii][indx]<threshold)
            if all_nan:  # np.all(node_table2[-1,:]==np.nan):
                if not(aggregate):  # new range started
                    starting_range = indx
                    aggregate = True
                    range_width = 1
                else: # already aggregating range
                    range_width = indx - starting_range + 1
            elif aggregate:  # stopped aggregation
                if (indx - starting_range - range_width < max_range):
                    range_width += 0
                else:
                    if range_width >= min_range:
                        finish_range = starting_range + range_width
                        ranges.append([starting_range, finish_range])
                    aggregate = False
    if aggregate and range_width>min_range:   # didn't finished aggregating
        ranges.append([starting_range, starting_range + range_width])
    return ranges



def load_data(file_path):
    global n_train
    global n_val
    global n_test
    global op_sheets
    global input_sheets
    global output_sheets
    global input_channels
    global output_channels
    global input_diff
    ranges, df = ExtractMultiData(file_path,100)  # Assuming max=100 is enough to represent all possible number of channels.
    # first calculate the days:
    # n_days = int(sum(1 for row in open(data_fname, 'r')) // day_slot)
    '''
    n_days = int(max_range // day_slot)
    n_train = int(n_train / 100 * n_days)
    n_val = int(n_val / 100 * n_days)
    n_test = n_days - n_train - n_val
    len_train, len_val = n_train * day_slot, n_val * day_slot
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    '''    # set the same datasets for all different inputs and outputs (based on all channels common ranges)
    # ranges for out of range including: [[5512, 18144], [18151, 22186], [22192, 24805], [36307, 40332], [40624, 42959]]
    #ranges = [[5493, 10553], [10838, 24805], [35173, 42959]]
    #ranges = [[18151, 22186], [22192, 24806], [36307, 40332]]
    ranges = [[5512, 18144], [18151, 22186], [22192, 24806]]
    maximum_range_indx = np.arange(0,len(ranges))
    maxs = [r[1] - r[0] for r in ranges]
    sr = sorted(zip(maxs, ranges, maximum_range_indx), reverse=True)
    ranges = [x for _,x,_ in sr]
    maxs = [x for x,_,_ in sr]
    train = df[ranges[0][0]:ranges[0][1],:,:]
    val = df[ranges[1][0]:ranges[1][1],:,:]
    test = df[ranges[2][0]:ranges[2][1],:,:]
    train = np.where(np.isnan(train), 0, train)
    val = np.where(np.isnan(val), 0, val)
    test = np.where(np.isnan(test), 0, test)
    n_train = maxs[0]
    n_val = maxs[1]
    n_test = maxs[2]
    # Use this loading for other stuff, such as prediction matrix:
    del df
    gc.collect()
    Update_sheets()
    gt = test[:,:,output_channels].copy()  # df[:,:,output_channels]
    #gt = test.copy().astype(float)  # ground truth data
    #gt = df[:, :, output_channels].astype(float)  # ground truth data
    #dff = np.squeeze(gtt)
    aaa = np.empty(gt.shape)
    aaa.fill(np.nan)

    return gt, aaa, train, val, test

def data_transform(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot    # number of complete days
    n_route = data.shape[1]   # number of sensors/stations
    #dim = data.shape[2]  # number of channels/features
    n_slot = day_slot - n_his - n_pred + 1 - n_pred_seq    # number of data samples for training per day, each sample=n_his+n_pred+n_pred_seq
    x = np.zeros([n_day * n_slot, len(input_sheets), n_his, n_route])  # [total samples, 1, total inputs for each sample, #sensors]
    y = np.zeros([n_day * n_slot, len(output_sheets), n_pred_seq, n_route])  # [total samples, 1=#output features, total outputs for each sample, #sensors]
    indexes = np.zeros([n_day * n_slot, 1])  # [total samples, 1] indexes of prediction of each slot

    # go over all data table (i=days, j=sensors)
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j   # current sample
            s = i * day_slot + j  # start of slot/sample
            e = s + n_his    # end of slot/sample (history part)
            # Correct (velocity prediction only, 1 channel input):
            xtemp = data[s:e, :, input_channels].reshape(n_his, n_route, len(input_sheets))  # for cases that len(input_sheets)=1
            xtemp = np.transpose(xtemp, (2, 0, 1))  # keep the data, only switch between order of dimensions!
            x[t, :, :, :] = xtemp#.reshape(dim, n_his, n_route)  # it's already (n_his, n_route), it just add 1 dimenstion
            ytemp = data[e+n_pred-1:e+n_pred-1+n_pred_seq,:,output_channels] # keep the data, only switch between order of dimensions!
            y[t, :, :, :] = ytemp.reshape(1, n_pred_seq, n_route)  # retrieve the predicted value (after the history part)
            '''
            # Wrong:
            x[t, :, :, :] = data[s:e].reshape(dim, n_his, n_route)  # it's already (n_his, n_route), it just add 1 dimenstion
            y[t, :, :, :] = data[e+n_pred-1:e+n_pred-1+n_pred_seq,:,0].reshape(1, n_pred_seq, n_route)  # retrieve the predicted value (after the history part)
            '''
            indexes[t] = e + n_pred - 1
    y = y.squeeze()
    return torch.Tensor(x), torch.Tensor(y), torch.Tensor(indexes)


def data_load_and_transform2(data, perm, n_his, n_pred, n_train, n_val, day_slot, device):
    # load data and transform, only with random splitting to train,val,test sets and non daily seperation.
    global dim
    n_slot = len(data) - n_his - n_pred + 1 - n_pred_seq    # total number of data samples, each sample=n_his+n_pred+n_pred_seq
    n_train = int(n_train/100*n_slot)
    n_val = int(n_val/100*n_slot)
    n_test = n_slot - n_train - n_val
    dim = data.shape[2]  # number of channels/features
    x_train = np.zeros([n_train, dim, n_his, n_route])  # [total samples, 1, total inputs for each sample, #sensors]
    y_train = np.zeros([n_train, 1, n_pred_seq, n_route])  # [total samples, 1, total outputs for each sample, #sensors], 1 value for prediction
    indexes_train = np.zeros([n_train, 1])  # [total samples, 1]
    x_val = np.zeros([n_val, dim, n_his, n_route])  # [total samples, 1, total inputs for each sample, #sensors]
    y_val = np.zeros([n_val, 1, n_pred_seq, n_route])  # [total samples, 1, total outputs for each sample, #sensors], 1 value for prediction
    indexes_val = np.zeros([n_val, 1])  # [total samples, 1]
    x_test = np.zeros([n_test, dim, n_his, n_route])  # [total samples, 1, total inputs for each sample, #sensors]
    y_test = np.zeros([n_test, 1, n_pred_seq, n_route])  # [total samples, 1, total outputs for each sample, #sensors], 1 value for prediction
    indexes_test = np.zeros([n_test, 1])  # [total samples, 1]
    # ------------------------- FIX THIS PART according to "data_transform" procedure if it works well:
    for i in range(n_slot):
         s = perm[i]  # start of slot/sample
         e = s + n_his    # end of slot/sample (history part)
         if (i<n_train):  # training set
             x_train[i, :, :, :] = data[s:e].reshape(dim, n_his, n_route)
             y_train[i, :, :, :] = data[e+n_pred-1:e+n_pred-1+n_pred_seq,:,1].reshape(1, n_pred_seq, n_route)
             indexes_train[i] = e + n_pred - 1
         elif (i<n_train+n_val):  # validation set
             x_val[i-n_train, :, :, :] = data[s:e].reshape(dim, n_his, n_route)
             y_val[i-n_train, :, :, :] = data[e+n_pred-1:e+n_pred-1+n_pred_seq,:,1].reshape(1, n_pred_seq, n_route)
             indexes_val[i-n_train] = e + n_pred - 1
         else:    # test set
             x_test[i-n_train-n_val, :, :, :] = data[s:e].reshape(dim, n_his, n_route)
             y_test[i-n_train-n_val, :, :, :] = data[e+n_pred-1:e+n_pred-1+n_pred_seq,:,1].reshape(1, n_pred_seq, n_route)
             indexes_test[i-n_train-n_val] = e + n_pred - 1
    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()
    return torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(indexes_train), torch.Tensor(x_val), torch.Tensor(y_val), torch.Tensor(indexes_val), torch.Tensor(x_test), torch.Tensor(y_test), torch.Tensor(indexes_test)


def predict_matrix(file_path):
    # initialize Prediction matrix with nans, so that we'd disregard missing (not-predicted) values in it when comparing to GT.
    _, df = ExtractMultiData(file_path,1)
    df = np.squeeze(df)
    aaa = np.empty(df.shape)
    aaa.fill(np.nan)
    return aaa



# Utils: ------------------------------------------------------------------------------------------


def Update_sheets():
    global op_sheets
    global input_sheets
    global output_sheets
    global input_channels
    global output_channels
    op_sheets = input_sheets + output_sheets
    dim = len(op_sheets)  # number of channels/features
    input_channels = np.zeros([len(input_sheets), 1], dtype=np.int32)
    for indx, curr_sheet in enumerate(input_sheets):
        input_channels[indx] = op_sheets.index(curr_sheet)
    output_channels = np.zeros([len(output_sheets), 1], dtype=np.int32)
    for indx, curr_sheet in enumerate(output_sheets):
        output_channels[indx] = op_sheets.index(curr_sheet)  # assuming only 1 output channel!
    input_channels = np.squeeze(input_channels)
    output_channels = np.squeeze(output_channels)


def CompareArrays(A,B):
    return np.array_equal(A,B) and np.array_equiv(A,B) and (A==B).all()


def AssignNames(nname):
    # NOTE! if you use is_pre_model=True, then we should have only one file for the current model!
    ii = 1
    while (os.path.exists('model_' + nname + str(ii) + '.pt')):
      ii += 1
    return nname, 'model_' + nname + str(ii) + '.pt'

def LoadModel(best_model):
    import os
    filename = fname
    # load best model from file:
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        best_model.load_state_dict(checkpoint['model_state_dict'])
    return best_model


def Evaluate(best_model):
    # load best model from file:
    #best_model = modeling()
    global Predictions
    best_model = LoadModel(best_model)
    # Evaluation
    # Eventually we do not save val for prediction, since it has different set..
    evaluate_model(best_model, loss, val_iter, False, scaler[output_channels])  # store all predictions of best model on validation set.
    l = evaluate_model(best_model, loss, test_iter, True, scaler[output_channels])
    MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler[output_channels])
    return l, MAE, MAPE, RMSE



def Plot_losses(file_name, name_list, loss_list): #*args):
    # plot lossess and their labels, respectively.
    #args_length = int(len(args)//2)
    loss_list_length = len(loss_list)
    fig = plt.figure(dpi=600)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    '''
    # finding minimum range if each method have different #epochs:
    min_range = 1e8
    for losss in range(args_length):
        if len(args[losss])<min_range:
            min_range = len(args[losss])    
    for losss in range(args_length):
        plt.plot(range(1, min_range + 1), args[losss][:min_range], label=args[losss+args_length])
    '''
    for losss in range(loss_list_length):
        plt.plot(range(1, len(loss_list[losss]) + 1), loss_list[losss], label=name_list[losss]) # + args_length])
    plt.legend()
    plt.show()
    if pd.notna(file_name):
        plt.savefig(file_name)

def Plot_prediction_serie(file_name, serie, nnode, gt):
    # plot results of predicting via training/validation/test, for some specific nnode:
    global Predictions
    fig = plt.figure(dpi=600)
    plt.xlabel("Time")
    plt.ylabel("Prediction")
    #_, df = ExtractMultiData(data_fname, 100)
    #gt = df[:,:,nsheet].astype(float)   # ground truth data
    plt.plot(range(1, len(serie) + 1), Predictions[serie,nnode], label='val/test prediction')
    plt.plot(range(1, len(serie) + 1), gt[serie,nnode], label='Ground Truth')
    plt.legend()
    if pd.notna(file_name):
         plt.savefig(file_name)
    plt.show()


def evaluate_model(model, loss, data_iter, is_store, scalerr):
    # Testing: return final loss
    global Predictions
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y, indx in data_iter:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x).view(len(x), -1)
            if is_store:
                indx1 = indx.numpy()
                indx1 = np.squeeze(indx1)
                indx1 = indx1.astype(int)
                #if np.all([ii in range(0,Predictions.shape[0]) for ii in indx1]):
                Predictions[indx1,:] = scalerr.inverse_transform(y_pred.cpu().numpy())
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]   # sum losses
            n += y.shape[0]             # sum number of losses
        return l_sum / n   # average loss for the whole evaluation


def evaluate_metric(model, data_iter, scalerr):
    # evaluate MAE, MAPE, RMSE
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y, indx in data_iter:
            x = x.to(device)
            y = y.to(device)
            y = scalerr.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scalerr.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE


def AppendFile(filename, *args):
    import os
    try:
        args = np.squeeze(args)
    except:
        pass
    args_length = int(len(args))
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        for indx in range(args_length):
            if indx % 2 == 0:  # add more
                checkpoint[args[indx]] = args[indx + 1]
    else:
        for indx in range(args_length):
            if indx == 0: # new
                checkpoint = {args[indx]: args[indx + 1]}
            elif indx % 2 == 0:  # add more
                checkpoint[args[indx]] = args[indx + 1]
    torch.save(checkpoint, filename)


def train_model():
    # Training and validating: return losses for all epochs
    train_loss_epoch = []
    valid_loss_epoch = []
    min_val_loss = np.inf
    print("Training model: ", name, ", total epochs:", epochs)
    for epoch in range(1, epochs + 1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y , indx in train_iter:
            x = x.to(device)
            y = y.to(device)
            #hidden = model.initHidden(batch_size)
            y_pred = model(x).view(len(x), -1)  # flatten the tensor to 1 dimension
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        if is_schedule:
            scheduler.step()   # to adjust learning rate
        val_loss1 = evaluate_model(model, loss, val_iter, False, scaler[output_channels])
        if val_loss1 < min_val_loss:
            min_val_loss = val_loss1
            model_state_dict = model.state_dict()
            AppendFile(fname, 'model_state_dict', model.state_dict())
        train_loss_epoch.append(l_sum / n)
        valid_loss_epoch.append(val_loss1)
        print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss1)

    #name_list.append(name)
    #train_loss.append(train_loss_epoch)
    #val_loss.append(valid_loss_epoch)
    #AppendFile(fname, 'train_loss_epoch', train_loss_epoch, 'valid_loss_epoch', valid_loss_epoch)
    # valid_loss_epoch = np.array(valid_loss_epoch).tolist()
    AppendFile(fname, 'valid_loss_epoch', valid_loss_epoch)
    return None#train_loss_epoch, valid_loss_epoch


# from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(modell):
    return sum(p.numel() for p in modell.parameters() if p.requires_grad)


def modeling():
    if name=="STGCN":
        mmodel = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob)
    if name=="STGCNb":
        mmodel = STGCNb(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob)
    if name=="STGCNc":
        mmodel = STGCNc(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob, input_diff)
    if name=="STGCNd":
        mmodel = STGCNd(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob, input_diff)
    if name=="STGCN2":
        mmodel = STGCN2(W, W.shape[0], 1, n_his, n_pred)
    if name=="GraphUNet":
        mmodel = GUNet(edge_index, n_route, n_his, n_pred_seq)
    if name=="TGCN1":
        mmodel = NetTGCN1(L, n_route, 1, 15, Ks, n_pred)  # for some reason Lk[0] is a bit better then L, which is what suppose to be...
    if name=="TGCN2":
        mmodel = NetTGCN2(L, n_route)
    if name=="TGCN3":
        mmodel = NetTGCN3(L, n_route)
    if name=="TGCN4":
        mmodel = NetTGCN4(L, n_route)
    if name=="LSTM":
        mmodel = LSTM(input_dim, hidden_dim, output_dim, output_last = True)
    if name=="Conv+LSTM":
        mmodel = ConvLSTM(input_dim, hidden_dim, output_dim, output_last=True)
    if name=="RNN":
        mmodel = RNN(input_dim, hidden_dim, output_dim, output_last = True)
    if name=="LSGC+LSTM":
        mmodel = LocalizedSpectralGraphConvolutionalLSTM(K, torch.Tensor(W), W.shape[0], Clamp_A=Clamp_A, output_last = True)
    if name=="SGC+LSTM":
        mmodel = SpectralGraphConvolutionalLSTM(K, torch.Tensor(W), W.shape[0], Clamp_A=Clamp_A, output_last = True)
    if name=="GAT":
        mmodel = GATNet(n_his, edge_index)
    if name=="GCN":
        mmodel = GCNNet(n_his, edge_index, edge_weight)
    if name=="GTN":
        mmodel = GraphTransformerNet(in_channels=n_his, hidden_channels=64, out_channels=1,
                                     edge_index=edge_index, edge_attr=edge_weight, dropout=drop_prob)
    #if name=="GC_LSTM":    #---- since we have no FFR
    #    model = GraphConvolutionalLSTM(K, torch.Tensor(W), FFR[back_length], W.shape[0], Clamp_A=Clamp_A, output_last = True)
    if name=="GGCN":
        mmodel = GGCN(torch.Tensor(W), train_tensor.size(3), args.num_classes,
                     [train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64],
                     args.feat_dims, args.dropout_rate)
    if is_pre_model:
        mmodel = LoadModel(mmodel)
    return mmodel.to(device)

def RunRandom(nnseed):
    torch.manual_seed(nnseed)
    torch.cuda.manual_seed(nnseed)
    np.random.seed(nnseed)
    random.seed(nnseed)
    torch.backends.cudnn.deterministic = True


# Main Program:  ------------------------------------------------------------------------------------------

import random
from sklearn.preprocessing import StandardScaler
from Models import *
from Utils import *

# Random Seed
nseed = 2333 # 2333
RunRandom(nseed)

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

# File Paths
python_path = "" #r"D:\Doctorate\Google Drive\My work\My Python"
matrix_name = "adj03_links_adj.npy" #"STGCN-PyTorch-master2/dataset/W_228.csv" #"adj01_links_adj.csv"  # Adjecancy matrix
# sequential data file: rows=time steps, columns=sensors in graph
data_path = "" #r"D:\Doctorate\Python\BTDATA\BT-Data"
data_name = "Data_2019.npy" #"BT_2019_07 - Copy.xlsx" #"STGCN-PyTorch-master2/dataset/V_228.csv" #"MAY_23_2020_to_PRESENT.xlsx" #"Data part1.csv"
data_fname = os.path.join(data_path,data_name)
#save_path = "best_model.pt"        # save and load best model after training
links_name = "adj04_links.npy"
links_fname = os.path.join(python_path,links_name)
sqr_sigma = 5e6
epsilon = 0.5
trans = [0,1,0,1] # to build adjacency matrix out of the links_fname file [1,0,0,0]. transform unweighted adjacency matrix into either weighted by distances [1,0,0], or by dijastra shortest path between each pair of nodes [0,1,0], or by exponential [0,0,1], or combination of those
is_binary = False # split binary codes in their 1-channel int form into multiple binary channels.
input_sheets = ["Velocity"]  # "Velocity", "Plans", "GreenTimes", "nVehs", "Plans_coded"
output_sheets = ["Velocity"]
is_pre_model = False # True = use previous model as weight initialization.

# Parameters
sample_secs = 60*5  # the tim e step of each sample, or the average value over what period of time the sample is.
day_slot = int(60*60/sample_secs*24) # number of slots/samples per day
n_train, n_val, n_test = 70, 15, 15  # partition dataset into train,val,test sets (in percentages).

epochs = 50
n_his = 12  # number of input time steps for each slot
n_pred = 3  # prediction of n_pred steps into the future, where 0 is the current step
n_pred_seq = 1 # prediction length (1 for a single value)
n_route = 228 # number of nodes in graph, will be redefined from the data.

batch_size = 50
is_permutate = False # Splitting data to train,val,test: True=permutate or daily-separated+ordered-split.
Pre_process = "Valid + some OutOfRange"   # "Valid only" "Valid+sat" "LeastNans" ""

#l, MAE, MAPE, RMSE, epochs, is_schedule, lr, name, fname, num_params = [[] for _ in range(10)]  # this format only! to seperate these variables!
#name = "STGCN"
def globals_tosave():
    return ['python_path', python_path, 'matrix_name', matrix_name, 'data_path', data_path, 'data_name', data_name,
                     'data_fname', data_fname, 'input_sheets', input_sheets, 'output_sheets', output_sheets, 'is_pre_model', is_pre_model,
                     'sample_secs', sample_secs, 'day_slot', day_slot, 'n_train', n_train, 'n_val', n_val, 'n_test', n_test, 'n_his', n_his,
                     'n_pred', n_pred, 'n_pred_seq', n_pred_seq, 'n_route', n_route, 'batch_size', batch_size, 'is_permutate', is_permutate,
                     'test loss', l, 'MAE', MAE, 'MAPE', MAPE, 'RMSE', RMSE, 'epochs', epochs, 'is_schedule', is_schedule, 'lr', lr,
                     'Model', name, 'Pre_process', Pre_process, 'test_iter', test_iter, 'num_params', num_params,
                     'fname', fname, 'links_fname', links_fname, 'sqr_sigma', sqr_sigma, 'epsilon', epsilon,
                     'is_binary', is_binary, 'trans', trans, 'Main_file.py', a1, 'Models.py', a2]
#AppendFile(fname, globals_tosave)  # test, delete after the test!
# Data Pre-processing ---------------------------------------------------------------------------

# for evaluation table, comparing different methods
#df2 = []
#op_sheets = ExtractSheets(data_fname) # operating sheets of data (input features, for each node)

Update_sheets()
# storing all val and test predictions, for comparison with ground truth
#Predictions = predict_matrix(data_fname)
#Predictions = None
#GT_data = None


def create_sets():
    global Predictions
    global GT_data
    global n_train
    global n_val
    global n_test
    global scaler
    global dim
    global op_sheets
    global input_sheets
    global output_sheets
    global input_channels
    global output_channels
    global input_diff
    if is_permutate:
        df = pd.read_csv(data_fname, header=None).values.astype(float)
        n_route = df.shape[1]  # number of sensors/stations

        df = scaler.fit_transform(df)
        n_slot = len(df) - n_his - n_pred + 1 - n_pred_seq # total number of data samples, each sample=n_his+n_pred+n_pred_seq
        perm = torch.randperm(n_slot)  # series of permutation/reordering of numbers 0,..,n_slot-1
        x_train, y_train, indexes_train, x_val, y_val, indexes_val, x_test, y_test, indexes_test = \
            data_load_and_transform2(df, perm, n_his, n_pred, n_train, n_val, day_slot, device)
    else:
        # Standardization
        GT_data, Predictions, train, val, test = load_data(data_fname)#, n_train, n_val)
        # AppendFile('GT_data.pt', 'GT_data', GT_data)
        dim = train.shape[2]
        scaler = []
        for i in range(dim):
            scaler.append(StandardScaler())  # transform data to its normialized form, mean=0, std = 1
        if train.size > 0:
            for i in range(dim):
                train[:,:,i] = scaler[i].fit_transform(train[:,:,i])
        if val.size > 0:
            for i in range(dim):
                val[:, :, i] = scaler[i].fit_transform(val[:, :, i])
        if test.size > 0:
            for i in range(dim):
                test[:, :, i] = scaler[i].fit_transform(test[:, :, i])

        # order the indexes (don't use if every set on it's own and not from one big set):
        #indexes_val = indexes_val + n_train * day_slot
        #indexes_test = indexes_test + n_train * day_slot + n_val * day_slot
    return train, val, test # DataLoader

def transpose_datasets(x2_train, x2_val, x2_test):
    x2_train = np.transpose(x2_train,(0,3,1,2)) #x2_train.squeeze().transpose(1, 2)
    x2_val = np.transpose(x2_val,(0,3,1,2)) #x2_val.squeeze().transpose(1, 2)
    x2_test = np.transpose(x2_test,(0,3,1,2)) #x2_test.squeeze().transpose(1, 2)
    # del train_iter; del val_iter; del test_iter
    return x2_train, x2_val, x2_test
def data_to_iter(x2_train, x2_val, x2_test):
    train_data = torch.utils.data.TensorDataset(x2_train, y_train, indexes_train)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(x2_val, y_val, indexes_val)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size)
    test_data = torch.utils.data.TensorDataset(x2_test, y_test, indexes_test)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size)
    return train_iter, val_iter, test_iter


# Preperation:

W0 = load_matrix(matrix_name)  # Adjacency matrix
n_route = W0.shape[0]
fea_size = W0.shape[1]  # features size = number of nodes enter input of RNN, each time step
# for RNN and LSTM models:
#inputs, labels = next(iter(train_iter))
#[batch_size, step_size, fea_size] = inputs.size()
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size
#name_list = []
#train_loss = []
#val_loss = []
import os, sys
f = open(sys.argv[0], mode='rb')
a1 = f.read()
f.close()
f = open(os.path.join(os.getcwd(),'Models.py'), mode='rb')
a2 = f.read()
f.close()


# Important: should group methods with the same preparation, to avoid errors...
#n_pred = 0  # predict the present, put this before "create_sets" procedure.
train, val, test = create_sets()
try:
    nuniques = len(np.unique(np.array(input_diff)))
except:
    nuniques = None
# Training & evaluate model ----------------------------------------------------------
# Transform Data
x_train, y_train, indexes_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val, indexes_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test, indexes_test = data_transform(test, n_his, n_pred, day_slot, device)
dim = len(input_sheets)




xx_train, xx_val, xx_test = x_train, x_val, x_test
# Get edge indices (non-zero connections) and edge weights
name, fname = AssignNames("GTN")  # model's name
Ks, Kt = 3, 3  # Kt=1D convolution/kernel size over the n_his sequnce of each node.
blocks = [[dim, 32, 64], [64, 32, 128]]  # sizes of feature maps in the STGCN model, for 2 ST-Conv blocks.
drop_prob = 0.4   # 0.5
lr = 1e-4  # learning rate
W = Adjacency_Matrix(W0, trans)  # Here, cause updates also L, Lk, as function of W and Ks
# edge_index = W.nonzero(as_tuple=False).t().contiguous()  # [2, E]
# edge_weight = W[edge_index[0], edge_index[1]]            # [E]
row, col = np.nonzero(W)
edge_index = torch.tensor([row, col], dtype=torch.long)
edge_weight = torch.tensor(W[row, col], dtype=torch.float32).unsqueeze(1)
# Loss & Model & Optimizer
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
batch_size = 50
train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# LR Scheduler
is_schedule = True
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

train_model()  # (name, epochs, train_iter, model, optimizer, scheduler)
# Load Best Model and Evaluate it:
best_model = modeling()   #(len(input_sheets),1)
l, MAE, MAPE, RMSE = Evaluate(best_model)
locals_tosave = ['Ks', Ks, 'Kt', Kt, 'blocks', blocks, 'drop_prob', drop_prob]
AppendFile(fname, *locals_tosave)
AppendFile(fname, *globals_tosave())


exit(0)


xx_train, xx_val, xx_test = x_train, x_val, x_test
name, fname = AssignNames("STGCN")  # model's name
Ks, Kt = 3, 3  # Kt=1D convolution/kernel size over the n_his sequnce of each node.
# Ks=kernel size of graph convolution, which determines the maximum radius of the convolution from central nodes.
#blocks = [[dim, int(32/len(input_sheets)), int(64/len(input_sheets))],
#          [int(64/len(input_sheets)), int(32/len(input_sheets)), int(128/len(input_sheets))]]  # sizes of feature maps in the STGCN model, for 2 ST-Conv blocks.
blocks = [[dim, 32, 64], [64, 32, 128]]  # sizes of feature maps in the STGCN model, for 2 ST-Conv blocks.
drop_prob = 0.4   # 0.5
# epochs = 50
lr = 1e-3  # learning rate
W = Adjacency_Matrix(W0, trans)  # Here, cause updates also L, Lk, as function of W and Ks
# Loss & Model & Optimizer
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
batch_size = 50
train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
# LR Scheduler
is_schedule = True
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

train_model()  # (name, epochs, train_iter, model, optimizer, scheduler)
# Load Best Model and Evaluate it:
best_model = modeling()   #(len(input_sheets),1)
l, MAE, MAPE, RMSE = Evaluate(best_model)
locals_tosave = ['Ks', Ks, 'Kt', Kt, 'blocks', blocks, 'drop_prob', drop_prob]
AppendFile(fname, *locals_tosave)
AppendFile(fname, *globals_tosave())
''''''





Plot_prediction_serie("predicted_vs_gt.png", np.arange(-1000, -100),13,GT_data) # see predictions of specific model, for specific node
print('Plot_prediction Finished!')





xx_train, xx_val, xx_test = transpose_datasets(x_train, x_val, x_test)
name, fname = AssignNames("GAT")  # model's name
lr = 0.005  # learning rate
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
batch_size = 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
train_model()
best_model = model
l, MAE, MAPE, RMSE = Evaluate(best_model)
locals_tosave = ['Ks', Ks, 'Kt', Kt, 'blocks', blocks, 'drop_prob', drop_prob]
AppendFile(fname, *locals_tosave)
AppendFile(fname, *globals_tosave())
''''''



name, fname = AssignNames("GCN")  # model's name
lr = 1e-2  # learning rate
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=lr)  # Only perform weight-decay on first convolution.
batch_size = 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
train_model()
best_model = model
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, *globals_tosave())
''''''


'''
name, fname = AssignNames("STGCN2")  # model's name
A_wave = get_normalized_adj(W)
A_wave = torch.from_numpy(A_wave)
W = A_wave.to(device)
epochs = 2
loss = nn.MSELoss()
x_train = np.transpose(x_train,(0,3,2,1))
x_val = np.transpose(x_val,(0,3,2,1))
x_test = np.transpose(x_test,(0,3,2,1))
train_data = torch.utils.data.TensorDataset(x_train,y_train, indexes_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val, indexes_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test, indexes_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)
model = modeling()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model()  # (name, epochs, train_iter, model, optimizer, scheduler)
# Load Best Model and Evaluate it:
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, globals_tosave)
'''

xx_train, xx_val, xx_test = x_train, x_val, x_test
name, fname = AssignNames("RNN")  # model's name
lr = 1e-5
#epochs = 100
is_schedule = False
scheduler = []
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)
batch_size = 50
train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)

train_model()
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, *globals_tosave())
''''''

name, fname = AssignNames("LSTM")  # model's name
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)

train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
train_model()
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, *globals_tosave())
''''''


name, fname = AssignNames("Conv+LSTM")  # model's name
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)

train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
train_model()
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, *globals_tosave())
''''''





name, fname = AssignNames("LSGC+LSTM")  # model's name
#epochs = 100
K = 64
Clamp_A = False
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)

train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
train_model()
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
locals_tosave = ['K', K, 'Clamp_A', Clamp_A]
AppendFile(fname, *locals_tosave)
AppendFile(fname, *globals_tosave())
''''''




name, fname = AssignNames("SGC+LSTM")  # model's name
K = 3
back_length = 3
Clamp_A = False
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)

train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
train_model()
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
locals_tosave = ['K', K, 'Clamp_A', Clamp_A, 'back_length', back_length]
AppendFile(fname, *locals_tosave)
AppendFile(fname, *globals_tosave())
''''''








'''
name, fname = AssignNames("GC+LSTM")  # model's name
model = modeling()
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)
train_model()
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, globals_tosave)
'''

'''
name, fname = AssignNames("GGCN")  # model's name, from "st-gcn-pytorch-master" folder
loss = nn.CrossEntropyLoss()
model = modeling()
optimizer = torch.optim.Adam(model.parameters(), lr = lr) #, betas=[args.beta1, args.beta2], weight_decay = args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.1)
train_model()
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, globals_tosave)
'''



'''
xx_train, xx_val, xx_test = transpose_datasets(x_train, x_val, x_test)
name, fname = AssignNames("TGCN1")  # model's name
Ks, Kt = 3, 3  # Kt=1D convolution/kernel size over the n_his sequnce of each node.
# Ks=kernel size of graph convolution, which determines the maximum radius of the convolution from central nodes.
epochs = 2
lr = 1e-3  # learning rate
# Graph
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)
L = torch.Tensor(L.astype(np.float32)).to(device)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
# Loss & Model & Optimizer
loss = nn.MSELoss()
model = modeling()
batch_size = 50
train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.005)
# LR Scheduler
is_schedule = True
''''''
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
train_model()  # (name, epochs, train_iter, model, optimizer, scheduler)
# Load Best Model and Evaluate it:
best_model = modeling()
l, MAE, MAPE, RMSE = Evaluate(best_model)
locals_tosave = ['Kt', Kt, 'Ks', Ks]
AppendFile(fname, locals_tosave)
AppendFile(fname, globals_tosave)
'''


# Predicting the input, 1D input:
'''
n_his = 1
n_pred = 0
n_pred_seq = 1
x_train, y_train, indexes_train, x_val, y_val, indexes_val, x_test, y_test, indexes_test = create_sets()
xx_train, xx_val, xx_test = transpose_datasets(x_train, x_val, x_test) #x_train, x_val, x_test
'''
'''
# Predicting the next time step based on previous ones:
xx_train, xx_val, xx_test = transpose_datasets(x_train, x_val, x_test)
name, fname = AssignNames("GraphUNet")  # model's name
#epochs = 100
lr = 1e-2  # learning rate
loss = nn.MSELoss()
model = modeling()
num_params = count_parameters(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
batch_size = 1
train_iter, val_iter, test_iter = data_to_iter(xx_train, xx_val, xx_test)
is_schedule = False

train_model()
best_model = model
l, MAE, MAPE, RMSE = Evaluate(best_model)
AppendFile(fname, *globals_tosave())
'''




# Show results ---- FROM NOW ON IN A "Results.py" FILE!
#Plot_losses(np.nan, train_loss_STGCN,valid_loss_STGCN, 'train loss STGCN','val loss STGCN')   # plot losess over epochs.
#print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)

#Model_names = ["STGCN","STGCN2","GraphUNet","TGCN1","TGCN2","TGCN3","TGCN4","LSTM","Conv+LSTM",
#               "RNN","LSGC+LSTM","SGC+LSTM","GAT","GCN","GC_LSTM","GGCN"]

# function to create table for all selected models:
# function to create plot for valid for epochs of all selected models:

#matplotlib.use('TkAgg')   # Since for some reason plot.show() don't work
#Plot_losses("val_losses_comparison.png", name_list, val_loss)

print("Finished!")