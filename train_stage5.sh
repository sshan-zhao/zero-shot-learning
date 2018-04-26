dataset=$1
cls=$2
threshold=$3
loss_weight=$4
gpu=$5
log_dir="./models/$dataset/$cls/log"
if [ ! -d $log_dir ]
then
	mkdir -p $log_dir
fi
python psr_train.py $dataset $cls $threshold $loss_weight $gpu 2>&1 | tee $log_dir/${threshold}_${loss_weight}_log.txt
