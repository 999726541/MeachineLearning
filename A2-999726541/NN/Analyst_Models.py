from util import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def stats_read(name):
    stats = Load(name)
    print('Validation_acc : {} \ntrain_acc : {} \nValidation_ce: {} \ntrain_ce: {}'.format(stats['valid_acc'][-1][1],
                        stats['train_acc'][-1][1],stats['valid_ce'][-1][1],stats['train_ce'][-1][1]))

def img_plot():
    model = Load("3.4_cnn_model_filter_45_45.npz")
    print(model["W1"].T.shape)
    for i in range(45):
        aa = model["W1"].T[i][0]
        #aa = np.array([aa])
        plt.subplot(9,5,i+1)
        #print(aa)
        zz = plt.imshow(aa)
    plt.show(zz)



#img_plot()

stats = Load('3.2_cnn_stats_eps_0.001.npz')
print(stats)
'''
'train_ce': train_ce_list,
'valid_ce': valid_ce_list,
'train_acc': train_acc_list,
'valid_acc': valid_acc_list
'''
DisplayPlot(stats['train_ce'],stats['valid_ce'],number=0,time=10000,ylabel='Cross Entropy')
DisplayPlot(stats['train_acc'],stats['valid_acc'],number=1,time=10000,ylabel='Accuracy')

#stats_read("3.3_cnn_stats_filter_4_8.npz")
print()
#stats_read("3.3_cnn_stats_filter_16_16.npz")
print()
#stats_read("3.3_nn_stats_hidden_48_96.npz")
print()
#stats_read("3.2_cnn_stats_eps_0.001.npz")
print()
#stats_read("3.2_cnn_stats_eps_1.0.npz")

