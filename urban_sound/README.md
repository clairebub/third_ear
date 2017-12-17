# UrbanSoundClassification Instruction

## Configuration:

**Always** install `anaconda` and set up an `virtualenv`.

Run the following command to install all dependencies.

```
pip install -r requirements.txt
```
If it shows the error about missing ffmpeg:
```
conda install -c conda-forge-ffmpeg
```

## Code Running:

Easy usage for inference:
```
usage: python dnn_modeling.py
```

To test new sound clip:
```
Copy the new .wav file into audio/ folder

Input test file name: [new .wav file name] after running "python dnn_modeling.py"
Type another .wav file name to test more

Type 'quit' to exit
```

Usage with arguments:
```
usage: python dnn_modeling.py [-h] [--learning_rate LEARNING_RATE]
                       [--batch_size BATCH_SIZE]
                       [--num_of_epochs NUM_OF_EPOCHS]
                       [--n_hidden_units_one N_HIDDEN_UNITS_ONE]
                       [--n_hidden_units_two N_HIDDEN_UNITS_TWO]
                       [--model MODEL] [--data_dir DATA_DIR]
                       [--feature_type FEATURE_TYPE]
                       [--checkpoint_dir CHECKPOINT_DIR]
                       [--trained_model TRAINED_MODEL] [--use_gpu [USE_GPU]]
                       [--nouse_gpu] [--training [TRAINING]] [--notraining]
```