'''
@author: Ishan Mohanty
Mini-Project: MLP Training with Back-Propagation for MNIST Classification using Numpy only for MNIST Dataset
'''

# Dependencies
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt


# Load Data
with h5py.File('mnist_traindata.hdf5','r') as f:
    xdata = f['xdata'][:]
    ydata = f['ydata'][:]


# Shuffle the data and split into train and test
np.random.seed(1)  
indices = np.random.permutation(xdata.shape[0])
train_idx = indices[:50000]
val_idx = indices[50000:]

xtrain = xdata[train_idx,:]
xval = xdata[val_idx,:]

ytrain = ydata[train_idx,:]
yval = ydata[val_idx,:]


#Function Definitions:

#ReLu
def relu(x):
    return (x)*(x > 0)


#Derivative of ReLu
def diff_relu(x):
    x[x>0]=1
    x[x<=0]=0
    return x


#Tanh 
def tanh(x):
    return np.tanh(x)


#Derivative of Tanh
def diff_tanh(x):
    result = 1 - (np.tanh(x)*np.tanh(x))
    return result


#Softmax
def softmax(x):
    result = np.zeros( shape=(x.shape[0],1) )
    frac = np.zeros( shape=(x.shape[0],1) )
    for i in range(x.shape[1]):
        num = np.exp(x[:,i]- np.max(x[:,i]))
        denom = np.sum(num)
        frac = num/denom
        frac = frac.reshape(x.shape[0],1)
        result = np.hstack( ( result, frac  ) )
    result = result[:,1:]
    return result


#FeedForward Process of the MLP
def forward(w,b,m,act_func):
    s = [np.array([0])]
    a = [np.array([0])]
    s.append(w[1]@m + b[1])
    if act_func[1] == 'relu':
        a.append( relu(s[1]) )
    elif act_func[1] == 'tanh':
        a.append( tanh(s[1]) )
    for i in range(2,len(layers_config)):
        s.append( w[i]@a[i-1] + b[i] )
        if act_func[i] == 'relu':
            a.append( relu(s[i]) )
        elif act_func[i] == 'tanh':
            a.append( tanh(s[i]) )
        elif act_func[i] == 'softmax':
            a.append( softmax(s[i]) )
    return a


#BackPropagation Functionality of the MLP
def backprop(a_l,y,w,start,end,m):
    all_d = []
    grad_w = []
    d_l = a_l[-1] - y[:,start*mini_bat_size:end*mini_bat_size] 
    all_d.append(d_l)
    grad_w.append(d_l@a_l[-2].T)
    j = 0
    for i in range( len(layers_config)-2 , 0 ,-1 ):
        if activation[i] == 'relu':
            d_l = (w[i+1].T@all_d[j])*diff_relu(a_l[i]) 
            all_d.append(d_l)    
        elif activation[i] == 'tanh':
            d_l = (w[i+1].T@all_d[j])*diff_tanh(a_l[i]) 
            all_d.append(d_l) 
        j+=1
        if i == 1:
            grad_w.append(all_d[j]@m.T)
        else:
            grad_w.append(all_d[j]@a_l[i-1].T)
    all_d.append(np.array([0]))
    grad_w.append(np.array([0]))
    all_d = all_d[::-1]
    grad_w = grad_w[::-1]
    return grad_w,all_d       


#Stochastic Gradient Descent Update of Weights
def update(w,b,dw,db):
    for i in range(len(layers_config)):
        w[i] = w[i] - ((eta/mini_bat_size)*dw[i])
        b[i] = b[i] - ((eta/mini_bat_size)*db[i])
    return w,b


#Validation Performance Calculation
def validation(x,y,w,b,act):
    act_layers = []
    act_layers = forward(w,b,x.T,act)
    a = act_layers[len(layers_config)-1]
    count = 0
    for i in range(x.shape[0]):
        pred_label = np.argmax(a[:,i],axis=0)
        true_label = np.argmax(y[i,:])
        if pred_label == true_label:
            count+=1
    acc = (count/x.shape[0])
    return acc


#Training Process of the MLP on 60,000   to find the best Model and Apply on the Test set of 10,000 images

#Initialize Hyper-Parameters
epochs = 50
mini_bat_size = 50
layers_config = [784,512,10]
activation = [0,'relu','relu','softmax']
eta = 0.1

#Initialize weights and biases
w = []
b = []
w.append(np.array([0]))
b.append(np.array([0]))
for i in range(1,len(layers_config)):
    w.append( np.random.randn(layers_config[i],layers_config[i-1]) * np.sqrt(2.0/layers_config[i-1]) )  #He parameter initialization. CS231N , Check this , L[i] changed to L[i-1]
    b.append( np.random.randn(layers_config[i],mini_bat_size) )    

t_acc = []
v_acc = []
sum_time = 0
for num_epochs in range(epochs): 
    start = time.time()
    if num_epochs == 20 or num_epochs == 40:
        eta = eta/2.0
    #shuffle train data
    shuff_train_idx = np.random.permutation(train_idx)
    xtrain = xdata[shuff_train_idx,:]
    ytrain = ydata[shuff_train_idx,:]
    #create mini batches
    mini_batches = []
    mini_range = (int)(xtrain.shape[0]/mini_bat_size)
    for k in range(mini_range):
        mini_batches.append(xtrain[k*mini_bat_size:(k+1)*mini_bat_size,:].T)
    
    s = 0
    e = 1
    for m in mini_batches:
        a_l = forward(w,b,m,activation)
        dw ,db = backprop(a_l,ytrain.T,w,s,e,m)
        s+=1
        e+=1  
        w,b = update(w,b,dw,db)
    
    end = time.time()
    for i in range(len(layers_config)):
        if i > 0:
            b[i] = np.sum(b[i],axis=1)/mini_bat_size
            b[i] = b[i].reshape(b[i].shape[0],1)
            
    print("Epoch : ", num_epochs+1)
    print("Execution time : ", (end-start)," secs")    
    sum_time+=(end-start)    
    train_acc = validation(xtrain,ytrain,w,b,activation)
    t_acc.append(train_acc*100)
    print("Training accuracy : ",train_acc*100 )
    val_acc = validation(xval,yval,w,b,activation)
    v_acc.append(val_acc*100)
    print("Validation accuracy : ", val_acc*100)
    print("############################################################################# \n")

print("Average execution time over ",epochs," epochs is: ", (sum_time/epochs) , " secs" )
print("Average train accuracy :", np.mean(t_acc) )
print("Average test accuracy :", np.mean(v_acc) )
plt.figure()
x = range(1,epochs+1)
plt.plot(x, t_acc, '-b', label='Training')
plt.plot(x, v_acc, '--r', label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title("Accuracy curves for train/val")
plt.legend()
plt.axvline(x=20,color='k')
plt.axvline(x=40,color='k')
plt.show()

#Get the best weights and bias from the best model
with h5py.File("HW3P2.hdf5", "w") as hf:
    hf.attrs['act'] = np.string_("tanh")
    hf.create_dataset('w1',data = w[1])
    hf.create_dataset('b1',data = b[1])
    hf.create_dataset('w2',data = w[2])
    hf.create_dataset('b2',data = b[2])