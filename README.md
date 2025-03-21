# Discover visual concepts hidden in internal representation of infant model
In this paper, we present an interdisciplinary study exploring this question: can a computational model that imitates the infant learning process develop broader visual concepts that extend beyond the vocabulary it has heard, similar to how infants naturally learn? To investigate this, we analyze a recently published model in Science by Vong et al., which is trained on longitudinal, egocentric images of a single child paired with transcribed parental speech. We introduce a training-free framework that can discover visual concept neurons hidden in the model's internal representations. Our findings show that these neurons can classify objects outside its original vocabulary.

## Dataset
You may simply run `dataset_setup.sh`
- Konkle Lab Dataset: [download here](http://olivalab.mit.edu/MM/archives/ObjectCategories.zip)
- Broden Dataset: [download here](http://netdissect.csail.mit.edu/data/broden1_227.zip)
 
### Env
```bash
conda create --name prob_cvcl python=3.8
conda activate prob_cvcl
pip install -r requirements.txt
python -m spacy download en_core_web_sm # download spacy model for CVCL
pip install -e .
```

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
or using bash script directly to label models using our settings:
```bash
.\scripts\neuron_labeling.sh
```

### 2. Run *NeuronClassifier*
Note that below process will generate trials for *n-way* classification in `json` format under `data/trials/` folder. 

#### Baseline
Run vanilla *n-way* classification:
```bash
.\scripts\baseline_n_way.sh 
```

#### Neuron-based Classification
To reproduce our study using neurons for classification inside representations:
```bash
.\scripts\neuron_n_way.sh
```

### 3. CKA Layer Analysis
Comparing infant CVCL model with ResNext and CLIP models using CKA:
```bash
python -m src.compute_cka --batch_size 512 --model1 resnext
python -m src.compute_cka --batch_size 512 --model1 clip-res
```

## Figures 


## Repo Structure

## Citation
If you find our study helpful, please consider citing:
```bibtex
@inproceedings{ke2025discovering,
  author = {Xueyi Ke and Satoshi Tsutsui and Yayun Zhang and Bihan Wen},
  title = {Discovering Hidden Visual Concepts Beyond Linguistic Input in Infant Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
  note = {To appear}
}
```