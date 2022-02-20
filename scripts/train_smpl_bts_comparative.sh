#EXPID=$1
#CONFIG=$2
PORT=${3:-23456}
HOST=$(hostname -i)

#python ./scripts/train_smpl.py \
#    --nThreads 4 \
#    --launcher pytorch --rank 0 \
#    --dist-url tcp://${HOST}:${PORT} \
#    --exp-id ${EXPID} \
#    --cfg ${CONFIG} --seed 123123

BASE_CONFIG=./configs/256x192_lr5e-4-res34_2x_mix_pretrain_batch16.yaml
BTS_CONFIG=./configs/256x192_lr5e-4-res34_2x_mix_fixed_bts_densenet161.yaml
BTS_CONCAT_ORIGINAL_CONFIG=./configs/256x192_lr5e-4-res34_2x_mix_fixed_bts_densenet161_concat_original.yaml
BTS_ADAPT_1X1_CONFIG=./configs/256x192_lr5e-4-res34_2x_mix_fixed_bts_densenet161_adapt1x1.yaml
BTS_ADAPT_1X1_CONCAT_ORIGINAL_CONFIG=./configs/256x192_lr5e-4-res34_2x_mix_fixed_bts_densenet161_adapt1x1_concat_original.yaml

# base experiment
for (( i=1; i<=1; i++ ))
do
    echo BASE $i
    python ./scripts/train_smpl.py \
        --nThreads 4 \
        --launcher pytorch --rank 0 \
        --dist-url tcp://${HOST}:${PORT} \
        --exp-id base_lr5e-4_epoch40 \
        --cfg $BASE_CONFIG --seed 123123
done

# bts. 2d fusion(hadamard, no extra concat)
for (( i=1; i<=1; i++ ))
do
    echo BTS_NO_CONCAT $i
    python ./scripts/train_smpl.py \
        --nThreads 4 \
        --launcher pytorch --rank 0 \
        --dist-url tcp://${HOST}:${PORT} \
        --exp-id bts_no_concat_lr5e-4_epoch40 \
        --cfg $BTS_CONFIG --seed 123123
done

# bts. 2d fusion(hadamard, concat original feature)
for (( i=1; i<=1; i++ ))
do
    echo BTS_CONCAT_ORIGINAL $i
    python ./scripts/train_smpl.py \
        --nThreads 4 \
        --launcher pytorch --rank 0 \
        --dist-url tcp://${HOST}:${PORT} \
        --exp-id bts_concat_original_lr5e-4_epoch40 \
        --cfg $BTS_CONCAT_ORIGINAL_CONFIG --seed 123123
done

# bts. 2d fusion(hadamard, concat depth feature)

# bts. 2d fusion(hadamard, use full depth features. no extra concat)

# bts. 2d fusion(hadamard, use full depth features. concat original and depth features)

# bts. 2d fusion(hadamard. no concat). 1x1 conv adaptation
for (( i=1; i<=1; i++ ))
do
    echo BTS_ADAPT_1X1 $i
    python ./scripts/train_smpl.py \
        --nThreads 4 \
        --launcher pytorch --rank 0 \
        --dist-url tcp://${HOST}:${PORT} \
        --exp-id bts_no_concat_adapt_1x1_lr5e-4_epoch40 \
        --cfg $BTS_ADAPT_1X1_CONFIG --seed 123123
done

# bts. 2d fusion(hadamard. concat original feature). 1x1 conv adaptation
for (( i=1; i<=1; i++ ))
do
    echo BTS_ADAPT_1X1 $i
    python ./scripts/train_smpl.py \
        --nThreads 4 \
        --launcher pytorch --rank 0 \
        --dist-url tcp://${HOST}:${PORT} \
        --exp-id bts_concat_original_adapt_1x1_lr5e-4_epoch40 \
        --cfg $BTS_ADAPT_1X1_CONCAT_ORIGINAL_CONFIG --seed 123123
done
