dataset=$1
cls=$2
iter=$3
test_file="data/$dataset/${cls}_test"
job_name="PSR_0.1_0.01"
train_file="data/$dataset/${cls}_train/images"
result_file="$job_name-${cls}.txt"
model_file="models/$dataset/${cls}/${job_name}"

if [ ! -d $save_file ]
then
	mkdir -p $save_file
fi

python psr_detect.py --gpu_id 0 --model_def $model_file/deploy.prototxt \
		--model_weights $model_file/snapshot/${job_name}_iter_$iter.caffemodel \
		--images_file $test_file --result_file $result_file --feat_file norm_feat.txt\
		--attribute_file $train_file/norm_unknown_attrs.txt \



