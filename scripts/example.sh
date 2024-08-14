GPUID=0
OUTDIR=outputs/RAF-DB/
REPEAT=1
mkdir -p $OUTDIR
python3 -u ../main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --no_class_remap --force_out_dim_fer 7 --force_out_dim_au 12 --force_out_dim_av 2 --schedule 1 --batch_size 24 --model_type custom_cnn --model_name Net --agent_type customization --lr 0.0001 --reg_coef 1 10 100 --train_aug
