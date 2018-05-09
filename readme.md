# [![result](Media/testing.png)](https://github.com/NH1900)
## Catalog
* [Background](#background)
* [Usage](#usage)
* [Content](#content)
    * Siganl Processing
    * Data Preprocessing
    * Deep Learning Modal
* [Future](#future)

# Background
Using deep learning method to process audio data after MFCC algoritm to classify different instruments.
# Usage
## Installation
Before installing Keras, please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend.

## TensorFlow installation instructions.
Theano installation instructions.
CNTK installation instructions.
You may also consider installing the following optional dependencies:

cuDNN (recommended if you plan on running Keras on GPU).
HDF5 and h5py (required if you plan on saving Keras models to disk).
graphviz and pydot (used by visualization utilities to plot model graphs).
Then, you can install Keras itself. There are two ways to install Keras:

## Install Keras from PyPI (recommended):
sudo pip install keras
If you are using a virtualenv, you may want to avoid using sudo:

pip install keras
Alternatively: install Keras from the GitHub source:
First, clone Keras using git:

git clone https://github.com/keras-team/keras.git
Then, cd to the Keras folder and run the install command:

cd keras
sudo python setup.py install
# Content
Part of code will be showed below to help you guys understand what I have done.
## Singnal processing
### Spectrum
In the audio spectrum after doing Short-Time Fourier Transform, for a piece of 
audio to divide this into many frame, spectrogram can be obtained in which 
formant can be found more readily.So before processing the audio pieces,just 
transfer the audio pieces into spectrogram.
```MATLAB
file1 = dir('NoteSamples\BbClar\*.wav');
lb = length(file1);
windowham = hamming(2048);
Bpoint = zeros(length(file1), 3);
for i = 1:length(file1)
    temp = wavread(['NoteSamples\BbClar\',file1(i).name]);
    a = 1;
    b = 2048;
    for j = 1:5
        frame = temp(a + (10+j-1-1)*2048: b + (10+j-1-1)*2048, 1);
        frame = frame.*windowham;
        if ((i == 1) && (j == 1))
            Bb = frame;
        else
            Bb = [Bb frame];
        end
        C = myCeps(frame, 21, 2048);
        if ((i == 1) && (j == 1))
            BbCldat = C;
        else 
            BbCldat = [BbCldat  C];
        end
    end
end
```
### Mel Filters
Since the frequency component are concentrate on human range,Mel filters are used to process the signal
```MATLAB
maxmelf = 2595*log10(1+22050/700);
sidewidth = maxmelf/(22+1);

index = 0:21;
filterbankcenter = (10.^(((index+1)*sidewidth)/2595)-1)*700;
filterbankstart = (10.^((index*sidewidth)/2595)-1)*700;
filterbankend = (10.^(((index+2)*sidewidth)/2595)-1)*700;
filterbankcenter = floor(filterbankcenter*1024/22050);
filterbankstart = floor(filterbankstart*1024/22050);
filterbankend = floor(filterbankend*1024/22050);
filterbankstart(1) = 1;
filtmag = zeros(1024, 1);
tbfCoef = zeros(22, 1);
```
### Log
Transfer the amplitude into DB:
### Cepstral
Transferring spectrogram into cepstral to separate spectral envelope and spectral details.Formants are more obvious in envelope.So that is the reason we did this. The way to realize this is doing the DCT(Discrete Consine Transform similar to IFFT).
```MATLAB
for i = 1:22
    for j = filterbankstart(i):filterbankcenter(i)
        filtmag(j, 1) = (j-filterbankstart(i))/(filterbankcenter(i)-filterbankstart(i));
    end
    for j = filterbankcenter(i):filterbankend(i)
        filtmag(j, 1) = (filterbankend(i)-j)/(filterbankend(i)-filterbankcenter(i));
    end
	%spectragram after filter
    tbfCoef(i, 1) = sum(FR(filterbankstart(i):filterbankend(i)).*filtmag(filterbankstart(i):filterbankend(i)));
end

tbfCoef = log(abs(tbfCoef));    
cc = dct(tbfCoef);
cc = cc(1:p, 1);
```
### Output
What the algorihm get is a 
21 * 1 vector for a frame. 
## Data preprocessing
### One Hot Key
In this project, there are three instruments needed to be classified.But they are three different strings. If we want to input those labels into LSTM, we have to transfer these into numbers. For those three lables(Flute, Clarinet and Trumpet),an unkown instrument has the same possibility to be any of them. So using one hot key encoding to ensure the euqal possibility(same distance to each other(vectors))
```python
def one_hot(label,instance_size,onehot_number):#onehot number equals to the number of classes
    onehot_matrix = np.zeros((instance_size,onehot_number))
    for i in range(instance_size):
        if label[i] == 1:
            onehot_matrix[i,0] = 1
        elif label[i] == 2:
            onehot_matrix[i,1] = 1
        elif label[i] == 3:
            onehot_matrix[i,2] = 1    
    return onehot_matrix
```
### Normalization
What normalization did is contract the range of input. In this project, we get a 21 * 5 matrix for a frame of audio. Values in each matrix varies. What LSTM need to obtain are some coefficients to calculate predicted result. Normalization help us limit values of features into a small range which can enhance the efficiency of learning. In this project minmax normalization is implemented.
```python
def normalize(data):
    len_batch,lenx,leny = data.shape
    for i in range(len_batch):
        for j in range(leny):
            (data[i,:,j] - data[i,:,j].mean()) / data[i,:,j].var()
    return data
```
### Splitting and Randomly distrubition
To minimize the influence of the sequence of input matrix, we use a random seed to shuffle input data.
Before we use the modal to testing dataset training dataset has to be splitted into training and tuning dateset to give a feedback to modal.
```python
def data_splitting(data):
    lenx,leny = data.shape
    length = leny - 900
    for i in range(length):
        data = np.delete(data,-1,1)
    return data

def batch_transpose(data):
    len_batch,lenx,leny = data.shape
    new_data = np.zeros(((len_batch,leny,lenx)))
    for i in range(len_batch):
        new_data[i,:,:] = np.transpose(data[i,:,:])
    return new_data
```
## Deep learning modal
```python
#normalize the data
train_norm = normalize(train_data_transpose)
tune_norm = normalize(tune_data_transpose)

#one hot key
train_onehot = one_hot(train_label,len(train_label),3)
tune_onehot = one_hot(tune_label,len(tune_label),3)

#build RNN model
model = Sequential()

##RNN cell
model.add(LSTM(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim = CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
# Future
In the future work, I plan to change the MFCC algoirhtm to differential MFCC algorithm.
Then training data set is not big enough. So collecting more data from Internet is also neccessary.


