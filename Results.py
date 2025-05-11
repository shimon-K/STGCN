import os
import re
import numpy as np
import torch
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from openpyxl import load_workbook

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd



def Results(folder_path, is_plot=False):
    if folder_path == '':
        root = tk.Tk()
        root.withdraw()
        # This program print comparison table + plot results for epochs, in a given folder of models:
        folder_path = filedialog.askdirectory()
        # r'C:\Users\shiman\Google Drive\My work\BTDATA 2020-06\best models data\2020-08-27'  # directory with all models of interest

    df = []
    df2 = []  # table of comparison.
    # more parameters to choose for comparison:
    # 'python_path', 'matrix_name', 'data_path', 'data_name', 'data_fname', 'input_sheets', 'output_sheets', 'is_pre_model', 'sample_secs',
    # 'day_slot', 'n_train', 'n_val', 'n_test', 'n_his', 'n_pred', 'n_pred_seq', 'n_route', 'batch_size', 'is_permutate', 'test loss', 'MAE',
    # 'MAPE', 'RMSE', 'epochs', ... and locals! see Main file..
    #                      'is_schedule', is_schedule, 'lr', lr, 'name'
    # required_items = ['Model', 'test loss', 'MAE', 'MAPE', 'RMSE', 'epochs']
    is_add_locals = False  # whether to add local parameters to items
    # prepare for plot:
    Font_Size = 16
    max_epochs = 0  # if 0=No limit, otherwise limit number of epochs shown in plot.
    figname = "results.png"  # file to save the plot to.
    max_plots_in_graph = 15  # any more than that will be plotted in new sub-figures
    max_plots_in_total = 20
    line_width = 4.0
    is_legend = True  # True=show legend
    is_min = False  # True=show only the common x-values to all plots, False=plot by the maximum value of all x_values
    is_add_average = False  # True=add average black line to compare between sub-plots

    # Create the list of all your folder content with os.listdir()
    folder_content = os.listdir(folder_path)
    folder_name = os.path.basename(folder_path)
    # Create a regex from the pattern
    # with the regex, you want to find 'vendor.' --> 'vendor\.' (. needs to be escaped)
    # then any alphanumeric characters --> '.*' (. is regex wildcard and .* means match 0 or more times any characters)
    # then you want to match .js at the end --> '\.js'
    nameparts = ['model_', '\.pt']
    regex_pattern = nameparts[0] + '.*' + nameparts[1]
    regex = re.compile(regex_pattern)

    # First go over all model files, gather all checkpoints
    fnames = []  # ['parameters']
    checkpoints = []
    # llabel = []
    arr = [[] for _ in range(max_plots_in_total)]  # assume no more than 10 seperate plots
    xlabel = [[] for _ in range(max_plots_in_total)]  # assume no more than 10 seperate plots
    ylabel = [[] for _ in range(max_plots_in_total)]  # assume no more than 10 seperate plots
    llabel = [[] for _ in range(max_plots_in_total)]  # assume no more than 10 seperate plots

    for path in folder_content:
        if regex.search(path):
            fnames.append(path)
            checkpoints.append(torch.load(os.path.join(folder_path, path)))
            # Getting all info for plotting results over epochs:
            '''
            if 'Model' in checkpoints[-1]:
                 llabel = checkpoints[-1]['Model']
            elif 'name' in checkpoints[-1]:
                 llabel = checkpoints[-1]['name']
            elif 'model_type' in checkpoints[-1]:
                 llabel = checkpoints[-1]['model_type']
            else:
                 llabel = ''
            '''
            # Gather all different data for plots:
            for indx, i in enumerate(['valid_loss_epoch', 'Average Reward', 'Average Cumulative Reward', 'Exploration Rate',
                                      'Average Reward Fixed', 'Average Cumulative Reward Fixed']):
                if i in checkpoints[-1]:
                    llabel[indx].append(path)
                    if 'array' in checkpoints[-1][i]:
                        arr[indx].append(checkpoints[-1][i]['array'])
                    elif type(checkpoints[-1][i]) is list:
                        arr[indx].append(checkpoints[-1][i])
                    if 'xlabel' in checkpoints[-1][i]:
                        xlabel[indx] = checkpoints[-1][i]['xlabel']  # don't append, assign only once, since it's the same for all
                    elif type(checkpoints[-1][i]) is list:
                        xlabel[indx] = 'epochs'
                    if 'ylabel' in checkpoints[-1][i]:
                        ylabel[indx] = checkpoints[-1][i]['ylabel']
                    elif type(checkpoints[-1][i]) is list:
                        ylabel[indx] = 'loss'
                    checkpoints[-1].pop(i, None)  # remove from checkpoint, to not include in comparison list we do below..
                    xlabel[indx] = 'Epochs'
                    ylabel[indx] = 'Loss'

            # Remove the comments, try to display the files:
            # a1 = checkpoints[-1]['Main_file.py']
            # a2 = checkpoints[-1]['Models.py']
            '''
            root = tk.Tk()
            # specify size of window.
            root.geometry("480x320")
            w = tk.Label(root, text='GeeksForGeeks', font="50")
            w.pack()

            # Create text widget and specify size.
            T = tk.Text(root, height=5, width=52)

            # Create label
            l = tk.Label(root, text="Main_file.py")
            l.config(font=("Courier", 14))

            Fact = """A man can be arrested in
            Italy for wearing a skirt in public."""

            # # Create button for next text.
            # b1 = tk.Button(root, text="Next")
            # # Create an Exit button.
            # b2 = tk.Button(root, text="Exit", command=root.destroy)


            l.pack(); T.pack()
            #b1.pack();  b2.pack()

            T.insert(tk.END, Fact)   # Insert The Fact.
            tk.messagebox.showinfo("Information", "Close the window (or leave it as is)\n and then close this message to continue the code..")
            '''
            # f = open(os.path.join(folder_path,os.path.splitext(path)[0]+'_Main_file.py'), mode='wb'); f.write(a1); f.close()
            # f = open(os.path.join(folder_path,os.path.splitext(path)[0]+'_Models.py'), mode='wb'); f.write(a2); f.close()

            for i in ['Main_file.py', 'Models.py', 'Model01b.py', 'Utils.py', 'model_state_dict', 'gtn_state_dict', 'last_optimizer_dict', 'valid_loss_epoch', 'train_loss_epoch',
                      'GT_data', 'test_iter', 'ya', 'y_preda', 'last_target_dict', 'best_target_dict', 'last_policy_dict', 'best_policy_dict',
                      'current_combinations', 'memory', 'predefined_ws2',
                      'train_dataset', 'val_dataset', 'test_dataset']:
                if i in checkpoints[-1]:
                    if re.search(r'.*\.py$|.*\.txt$', i): # either py or txt files
                        a1 = checkpoints[-1][i]
                        with open(os.path.join(folder_path, os.path.splitext(path)[0] + '_'+i), mode='wb') as f:
                            f.write(a1)
                    checkpoints[-1].pop(i, None)  # remove from checkpoint, to not include in comparison list we do below..


    for indx, i in enumerate(['valid_loss_epoch', 'Average Reward', 'Average Reward Fixed', 'Average Cumulative Reward', 'Exploration Rate']):
        if llabel[indx] != []:
            arr[indx] = [torch.tensor(xx) for xx in arr[indx]]
            if is_min:
                tot_length = min([xx.size(0) for xx in arr[indx]])
            else:
                tot_length = max([xx.size(0) for xx in arr[indx]])
                arr[indx] = [torch.nn.functional.pad(xx, (0,tot_length-xx.size(0)),mode='constant', value=2.0) if tot_length>xx.size(0) else xx for xx in arr[indx]]

            if is_add_average:
                avg_plot = torch.mean(torch.stack(arr[indx], dim=1), axis=1)
            # Save data in tables in Excel files:
            writer = pd.ExcelWriter(os.path.join(folder_path,'results' + str(indx) + '.xlsx'), engine='openpyxl')
            matx = torch.stack(arr[indx]).transpose(0, 1).numpy()
            if matx.shape[0]<1048576 and matx.shape[1]<16384:
                pd.DataFrame(matx).to_excel(writer, sheet_name=i, header=llabel[indx], index=list(range(matx.shape[0])))
                writer._save()
                writer.close()
            # plot numerous number of graphs:
            nplots = torch.floor(torch.tensor(len(llabel[indx]) / max_plots_in_graph))
            if nplots < torch.tensor(len(llabel[indx]) / max_plots_in_graph): nplots += 1
            x_plots = torch.floor(torch.sqrt(nplots))
            y_plots = nplots // x_plots
            if nplots // x_plots < nplots / x_plots:
                y_plots += 1

            fig, axxx = plt.subplots(int(x_plots), int(y_plots), figsize=(6, 6), sharex='all', sharey='all', dpi=600)
            plt.xlabel(xlabel[indx], fontsize=Font_Size)
            plt.ylabel(ylabel[indx], fontsize=Font_Size)
            plt.xticks(fontsize=Font_Size)
            plt.yticks(fontsize=Font_Size)
            plt.subplots_adjust(hspace=.0, wspace=.0)

            # gs = fig.add_gridspec(int(x_plots), int(y_plots), hspace=0, wspace=0)
            # ax_x, ax_y = gs.subplots(sharex='col', sharey='row')
            # axx = np.concatenate((ax_x,ax_y))
            # axxx = axxx.reshape((1, -1))
            last_indxx, last_indyy, hhh_old, z_order = -1, -1, -1, 0
            for i in range(len(llabel[indx])):
                curr_plot = i // max_plots_in_graph
                indxx = int(curr_plot // y_plots.numpy())
                indyy = int(curr_plot % y_plots.numpy())
                if x_plots == y_plots == 1:
                    hhh = axxx
                elif x_plots == 1:
                    hhh = axxx[indyy]
                else:
                    hhh = axxx[indxx, indyy]
                if last_indxx!=indxx or last_indyy!=indyy:
                    if is_add_average:
                        if max_epochs == 0:
                            hhh.plot(range(1, len(avg_plot) + 1), avg_plot, color = 'black',label='avg', linewidth=line_width, zorder=100)
                        else:
                            hhh.plot(range(1, max_epochs + 1), avg_plot[:max_epochs], color = 'black',label='avg', linewidth=2.0, zorder=100)
                if max_epochs == 0:
                    hhh.plot(range(1, len(arr[indx][i]) + 1), arr[indx][i], #label=llabel[indx][i],
                                     linewidth=line_width, zorder=z_order)
                else:
                    hhh.plot(range(1, max_epochs + 1), arr[indx][i][:max_epochs], #label=llabel[indx][i],
                                    linewidth=2*line_width, zorder=z_order)
                # if not(i%max_plots_in_graph==0): plt.setp(axx[indx].get_xticklabels(), visible=False)
                last_indxx = indxx
                last_indyy = indyy
                z_order += 1


            for ax in fig.get_axes():
                if not is_legend:
                    ax.get_legend().remove()
                else:
                    ax.legend(prop={'size': int(Font_Size/np.ceil(len(llabel[indx])/max_plots_in_graph))})  #(loc='center right', bbox_to_anchor=(1.0, 0.8), prop={'size': Font_Size})
                    ax.label_outer()
                    ax.grid()




            # for plot location choose: best, upper right, upper left, lower left, lower right,
            # 	right, center left,	center right, lower center,	upper center, center

            # plt.legend(loc = 'center right', bbox_to_anchor=(1.0,0.8), prop={'size': Font_Size})
            # plt.xticks(fontsize=Font_Size)
            # plt.yticks(fontsize=Font_Size)

            # matplotlib.use('TkAgg')   # Since for some reason plot.show() don't work
            # Important: save to file before plt.show, since it creates new blank image.
            fname = os.path.join(folder_path, 'results' + str(indx) + '.png')
            # os.chdir(folder_path)
            if pd.notna(fname):
                plt.savefig(fname)

            if is_plot:
                plt.show()

    # Then search for identical parameters and different ones, to put in a comparison table
    num_checkpoints = len(checkpoints)
    first_chk = checkpoints[0]
    total_list = {}
    identical_list, different_list = [[] for _ in range(2)]
    for key in first_chk:
        total_list[key] = [None] * num_checkpoints  # create list of Nones (nans are for array)
        total_list[key][0] = first_chk[key]
    # unappeared_frstchk_list = first_chk.copy()
    checkpoints.pop(0)  # remove 1st checkpoint from the list
    for indx, checkpoint in enumerate(checkpoints):
        #    intersection = [k for k in first_chk if k in checkpoint]
        for key in checkpoint:
            if key in total_list.keys():
                total_list[key][indx + 1] = checkpoint[key]
            else:
                total_list[key] = [None] * num_checkpoints  # create list of Nones (nans are for array)
                total_list[key][indx + 1] = checkpoint[key]

    keys_delete = []
    for key in total_list:
        number_nones = sum(x is None for x in total_list[key])
        if (number_nones == num_checkpoints - 1) and not (is_add_locals) and (num_checkpoints > 2):
            keys_delete.append(key)

    for key in keys_delete:
        del total_list[key]  # do not present single items in table

    for key in total_list:
        if key=='predefined_ws':
            continue #print(key)
        print(key)
        if type(total_list[key][0])==torch.Tensor:
            if len(total_list[key])==1:
                df.append('some value1')
                identical_list.append(key)
            elif all([(total_list[key][xx]==total_list[key][xx+1]).all() for xx in range(len(total_list[key])-1)]):
                df.append('some value2')
                identical_list.append(key)
            else:
                df2.append('some value2')
                different_list.append(key)
        elif total_list[key].count(total_list[key][0]) == len(total_list[key]):  # all items are identical in this key list
            # if df==[]:
            #    df.append(key)
            df.append(total_list[key][0])
            identical_list.append(key)
        else:
            # if df2==[]:
            #    df2.append(key)
            df2.append(total_list[key])
            different_list.append(key)

    # for table:
    writer = pd.ExcelWriter(os.path.join(folder_path, folder_name + '.xlsx'), engine='openpyxl')
    if not (identical_list == []):
        df = pd.DataFrame(df, index=identical_list)
        print(df)
        df.to_excel(writer, sheet_name="Identical")
    if not (different_list == []):
        df2 = pd.DataFrame(df2, columns=fnames, index=different_list)
        print(df2)  # full table of comparison between models
        df2.to_excel(writer, sheet_name="Different")

    writer._save()
    writer.close()

    print('Finished!')




if __name__ == '__main__':
    Results('', True)


