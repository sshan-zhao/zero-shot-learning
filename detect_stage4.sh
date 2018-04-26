dataset=$1
cls=$2
iter=$3
dataset_type=$4
resize=224
test_file="data/$dataset/${cls}_$dataset_type"
job_name="CLS"
pred_bbox_file="detect_results/$dataset/${cls}_test/SSD_300x300_iter_120000_test"
train_file="data/$dataset/${cls}_train/images"
#save_file="detect_results/$dataset/${cls}_test/${job_name}_${resize}x${resize}_iter_${iter}"
model_file="models/$dataset/${cls}_$resize/${job_name}_${resize}x${resize}"

if [ ! -d $save_file ]
then
	mkdir -p $save_file
fi

python cls_detect.py --gpu_id 0 --model_def $model_file/deploy.prototxt \
		--model_weights $model_file/snapshot/${job_name}_${resize}x${resize}_iter_$iter.caffemodel \
	 	--mean_file lmdb-dataset/$dataset/${cls}_300/bgr_mean.txt \
		--image_resize $resize --images_file $test_file --train_file $train_file\
		--attribute_file attributes_per_class.txt --result_file ${cls}_${iter}.txt\
	        --pred_bbox_file $pred_bbox_file --dataset $dataset_type --label_file labels.txt



