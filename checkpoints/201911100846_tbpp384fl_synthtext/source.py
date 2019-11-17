# In[0]


# In[1]
get_ipython().run_line_magic('matplotlib', 'nbagg')
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import pickle

import tensorflow.keras.backend as K
from model import TBPP384
from tbpp_prior import PriorUtil
from tbpp_data import InputGenerator
from tbpp_training import TBPPFocalLoss
from utils.model import load_weights
from utils.training import Logger

# In[2]
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# In[3]
from data_synthtext import GTUtility
with open('gt_util_synthtext_seglink.pkl', 'rb') as f:
    gt_util = pickle.load(f)

gt_util_train, gt_util_val = gt_util.split(0.9)

# In[4]
# TextBoxes++
K.clear_session()
model = TBPP384(softmax=False)
weights_path = None

experiment = 'tbpp384fl_synthtext'

# In[5]
prior_util = PriorUtil(model)

# In[6]
epochs = 10
initial_epoch = 0
batch_size = 32

gen_train = InputGenerator(gt_util_train, prior_util, batch_size, model.image_size)
gen_val = InputGenerator(gt_util_val, prior_util, batch_size, model.image_size)


checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + experiment
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())

optim = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
# optim = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)

# weight decay
regularizer = tf.keras.regularizers.l2(5e-4) # None if disabled
#regularizer = None
for l in model.layers:
    if l.__class__.__name__.startswith('Conv'):
        l.kernel_regularizer = regularizer

loss = TBPPFocalLoss(lambda_conf=10000.0, lambda_offsets=1.0)

model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)

print(checkdir.split('/')[-1])

history = model.fit_generator(
        gen_train.generate(),
        steps_per_epoch=int(gen_train.num_batches/4), 
        epochs=epochs, 
        verbose=1, 
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
            Logger(checkdir),
            #LearningRateDecay()
        ], 
        validation_data=gen_val.generate(), 
        validation_steps=gen_val.num_batches, 
        class_weight=None,
        max_queue_size=1000, 
        workers=12, 
        use_multiprocessing=True, 
        initial_epoch=initial_epoch, 
        #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
        )

