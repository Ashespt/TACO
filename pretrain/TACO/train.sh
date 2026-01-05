export CUDA_VISIBLE_DEVICES=0
export LOGDIR='./runs_all/test'
export NUM_NODES=1
export SPACE_SIZE=96
export ALPHA_DOMAIN=0
export ALPHA_ADV=0
export MASTER_PORT=13227
export FEATURE_SIZE=48
export FEATURE_DIM=768
export NUM_DOMAINS=3
export ROI_LARGE=96
mkdir -p $LOGDIRf
cp train.sh "$LOGDIR/train.sh"
python -m torch.distributed.launch \
--nproc_per_node $NUM_NODES --master_port $MASTER_PORT main.py \
--alpha_adv $ALPHA_ADV --roi_large $ROI_LARGE \
--data_type 'Ours' \
--use_last_layer \
--num_geo_layer -1 \
--batch_size=2 --num_steps=50000 --lrdecay --eval_num=1000 --lr=3e-4 --decay=0.1 --alpha_domain $ALPHA_DOMAIN \
--feature_size $FEATURE_SIZE --feature_dim $FEATURE_DIM --in_channels 1 \
 --logdir $LOGDIR --num_domains $NUM_DOMAINS --roi_x $SPACE_SIZE --roi_y $SPACE_SIZE --roi_z $SPACE_SIZE \
--noamp 