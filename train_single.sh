j=$1
MODEL=DPT
DATASET=fgvc_aircraft # [eurosat caltech101 oxford_flowers food101 fgvc_aircraft]
DATADIR=FGVC/fgvc-aircraft-2013b #[EuroSAT Caltech101 Flowers102 Food101 FGVC/fgvc-aircraft-2013b]
# DATASET=caltech101
# DATADIR=Caltech101
DATASET=eurosat
DATADIR=EuroSAT
# DATASET=stanford_cars
# DATADIR=StanfordCars
# DATASET=oxford_flowers
# DATADIR=Flowers102
# DATASET=dtd
# DATADIR=DTD
# DATASET=food101
# DATADIR=Food101
# DATASET=oxford_pets
# DATADIR=OxfordPets
# DATASET=imagenet
# DATADIR=ImageNet1K
# DATASET=sun397
# DATADIR=Sun397
# DATASET=ucf101
# DATADIR=UCF101

# CPN is the length of CAVPT
# BOTTOMLIMIT is the layers CAVPT inserted into. e.g. 8 means 8-12 layers are equipped with CAVPT. 12 means 12 layers are equipped with CAVPT.1 means every layer are equipped with CAVPT.
# C is our general knowledge guidence epoch.
# ALPHA is loss balancing parameter.

python submit_train.py --root ./datasets/${DATADIR} --seed $j \
--trainer $MODEL \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--output-dir ./output \
--config-file ./configs/trainers/VPT/vit_b32_deep.yaml \
TRAINER.COOP.N_CTX 16 \
TRAINER.COOP.CSC False \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 1 \
TRAINER.VPT.N_CTX 10 \
TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT 12 \
TRAINER.SELECTED_COVPT.CPN 10 \
OPTIM.LR 0.01 \
OPTIM.MAX_EPOCH 60 \
PRETRAIN.C 30 \
TRAINER.ALPHA 0.3

