import os, sys, inspect

# Torchereid import from folder above (not a package)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import torchreid

def show_available_models():
    torchreid.models.show_avai_models()

def load_data(source, target, transforms):

    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',  # root path to datasets
        sources=source,   # source (train) dataset
        targets=target,   # target (test) datase
        height=256,     # target image height
        width=128,      # target image width
        batch_size_train=32,    # number of images in a training batch
        batch_size_test=100,    # number of images in a test batch
        transforms=transforms  #  transformations from pytorch applied to model training
    )

    return datamanager

def build_model(datamanager, model):

    # A function wrapper for building a model.
    model = torchreid.models.build_model(
        name=model,    #  model name
        num_classes=datamanager.num_train_pids, # number of training identities
        loss='softmax', # loss function to optimize the model. Currently supports “softmax” and “triplet”
        pretrained=True # whether to load ImageNet-pretrained weights.
    )

    model = model.cuda()

    return model

def build_engine(datamanager, model): 

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    # A function wrapper for building a learning rate schedule
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    # Softmax-loss engine for image-reid.
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    return engine

def run_engine(engine, model_name):

    engine.run(
        save_dir='log/' + model_name,
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )


if __name__ == "__main__":
    """ This is executed when run from the command line """
   
    show_available_models()

    # Load market1501 dataset to train and market1501, cuhk03 to test
    # data augmentation random_flip and random_crop
    datamanager = load_data('market1501', 'market1501', ['random_flip', 'random_crop'])
    
    # Build model to train and evaluate 
    model = build_model(datamanager, 'resnet50')

    # Build engine with model and data
    engine = build_engine(datamanager, model)

    # Run engine
    run_engine(engine, 'resnet50')

    