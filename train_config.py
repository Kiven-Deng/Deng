"""All configurable training and model hyperparameters.

Edit this file only. Then run train.py directly.
"""

# ============================================================================
# FILES
# ============================================================================

DATASET_FILE = "data/ranking_dataset.pt"
CHECKPOINT_FILE = "checkpoints/mallows_transformer.pt"

# ============================================================================
# TRAIN / VALIDATION SPLIT
# ============================================================================

# Validation data is held out and never used by the optimizer.
VALIDATION_RATIO = 0.1
SPLIT_SEED = 42

# ============================================================================
# DATALOADER AND OPTIMIZATION
# ============================================================================

BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_BATCHES = None                 # None trains every batch in each epoch.

LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0               # 0 disables clipping.
RANDOM_SEED = 42

NUM_WORKERS = 0
SHUFFLE = True
DROP_LAST = False
DEVICE = "cuda"                    # "auto", "cpu", or "cuda".

# ============================================================================
# MODEL DIMENSIONS
# ============================================================================

# None infers the value from the dataset. If set, it must match the data.
EMBEDDING_DIM = None
NUM_HEADS = None                   # d=10 -> 5; d=128 -> 8.
DIM_FF = None                      # None -> 4 * embedding_dim.

NUM_LAYERS_ENCODER = 2
NUM_LAYERS_DECODER = 2
MAX_LEN_PE = 1000

# ============================================================================
# NORMALIZATION
# ============================================================================

NORM_TYPE = "batchnorm"            # "batchnorm" or "layernorm".
NORM_POSITION = "pre"              # "pre" or "post".

# ============================================================================
# POSITION MECHANISMS
# ============================================================================

USE_RAROPE = True
USE_ANLIBI = True
USE_ORIGINAL_DECODER_PE = False

RAROPE_A = 1.0
RAROPE_B = 1.0

ANLIBI_ALPHA_INIT = 0.01
ANLIBI_BETA_INIT = 0.01
ANLIBI_GAMMA = 1.0

# None inherits USE_RAROPE / USE_ANLIBI for that module.
USE_RAROPE_ENCODER = None
USE_RAROPE_DECODER_SELF = None
USE_RAROPE_DECODER_CROSS = None
USE_ANLIBI_ENCODER = None
USE_ANLIBI_DECODER_SELF = None
USE_ANLIBI_DECODER_CROSS = None

# ============================================================================
# MALLOWS / SUPERVISED TRAINING
# ============================================================================

MALLOWS_THETA = 1.0
MALLOWS_SEED = None
TRAINING_MODE = "teacher_forcing"
LOSS_TYPE = "cross_entropy"
PRESERVE_ORIGINAL_MASK_UPDATE = True

# ============================================================================
# CHECKPOINTS
# ============================================================================

SAVE_CHECKPOINT = True
CHECKPOINT_SAVE_EVERY_N_EPOCHS = 1

# ============================================================================
# STANDALONE VALIDATION
# ============================================================================

# Number of held-out rankings predicted and printed randomly by validation.py.
VALIDATION_SAMPLE_COUNT = 10

# None chooses different validation samples on every run; set an integer to
# make the sampled rankings reproducible.
VALIDATION_RANDOM_SEED = None

VALIDATION_BATCH_SIZE = 32
VALIDATION_OUTPUT_FILE = "outputs/validation/rankings.json"

# Rank-Biased Overlap persistence. Must satisfy 0 <= RBO_PERSISTENCE < 1.
RBO_PERSISTENCE = 0.9
