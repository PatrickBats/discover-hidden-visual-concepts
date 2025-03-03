# Discover visual concepts hidden in internal representation of infant model
In this paper, we present an interdisciplinary study exploring this question: can a computational model that imitates the infant learning process develop broader visual concepts that extend beyond the vocabulary it has heard, similar to how infants naturally learn? To investigate this, we analyze a recently published model in Science by Vong et al., which is trained on longitudinal, egocentric images of a single child paired with transcribed parental speech. We introduce a training-free framework that can discover visual concept neurons hidden in the model's internal representations. Our findings show that these neurons can classify objects outside its original vocabulary.

## Dataset
You can simply run `dataset_setup.sh`
- Konkle Lab Dataset: [download here](http://olivalab.mit.edu/MM/archives/ObjectCategories.zip)
- Broden Dataset: [download here](http://netdissect.csail.mit.edu/data/broden1_227.zip)
 
### Env
```bash
conda create --name prob_cvcl python=3.8
conda activate prob_cvcl
pip install -r requirements.txt
python -m spacy download en_core_web_sm

pip install -e .
```
### Baseline
To get vanilla n-way classification baseline, run `baseline.sh`.
## Run
### 1. Run Neuron Labeling
```bash
python describe_neurons.py \
  --similarity_fn soft_wpmi \
  --target_model cvcl-resnext \
  --target_layers vision_encoder.model.layer1,vision_encoder.model.layer2,vision_encoder.model.layer3,vision_encoder.model.layer4 \
  --d_probe imagenet_val \
  --concept_set data/baby+30k+imagenet.txt \
  --device cuda:0
```
or using bash script `neuron_labeling.sh`.

### 2. Using Neurons to Classificaiton
```bash
python trial.py \
  --model cvcl-resnext \
  --seed 0 \
  --device cuda:0 \
  --num_img_per_trial 4 \
```

### Directory Structure
```
.
├── README.md                  # Project overview and instructions
├── env.yml                    # Environment configuration
├── describe_neurons.py        # neuron labeling
├── trial.py                   # Trial classification script
│
├── AoA/                      # AoA analysis
├── data/                      # Concept sets
├── datasets/                  # Generated trial datasets
│   └── trials/               # Generated trial files
│
├── models/                    # Model implementations
│   └── zs_trial_predic.py    # Zero-shot trial prediction model
│
├── neuron_concepts/          # Neuron labeling results
│   ├── cvcl_*.csv           # CVCL model neuron descriptions
│   ├── clip_*.csv           # CLIP model neuron descriptions
│   ├── resnext_*.csv        # ResNext model neuron descriptions
│   └── dino_*_*.csv         # DINO model neuron descriptions
│
├── plots/                    # Analysis notebooks with plots
│
│
├── utils/                   # Utility functions
│
└── scripts/                # Shell scripts for batch processing
    ├── neuron_labeling.sh # Batch neuron labeling
    ├── neuron.sh         # Neuron-based prediction
    └── baseline.sh       # Baseline prediction
```
