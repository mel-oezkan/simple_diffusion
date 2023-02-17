# data
DATASET_NAME = "oxford_flowers102"
DATASET_REPS = 5
NUM_EPOCHS = 1  # train for at least 50 epochs for good results
IMAGE_SIZE = 64

# KID = Kernel Inception Distance, see related section
KID_IMAGE_SIZE = 75
KID_DIFFUSION_STEPS = 5
PLOT_DIFFUSION_STEPS = 20

# sampling
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95

# architecture
EMBEDDING_DIMS = 32
EMBEDDING_MAX_FREQUENCY = 1000.0
WIDTHS = [32, 64, 96, 128]
BLOCK_DEPTH = 2

# optimization
BATCH_SIZE = 64
EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
