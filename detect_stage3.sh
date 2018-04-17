dataset=$1
cls=$2
iter=$3
resize=300
test_file="data/$dataset/${cls}_test"
pred_bbox_file="detect_results/$dataset/${cls}_test/SSD_${resize}x${resize}_iter_$iter"
train_file="data/$dataset/${cls}_train/images"
save_file="detect_results/$dataset/${cls}_test/ZSL_ORI_ATTR_${resize}x${resize}_iter_$iter"
model_file="models/$dataset/${cls}_$resize/ZSL_ORI_ATTR_${resize}x${resize}"

if [ ! -d $save_file ]
then
	mkdir -p $save_file
fi

python zsl_detect.py --gpu_id 0 --model_def $model_file/deploy.prototxt \
		--model_weights $model_file/snapshot/ZSL_ORI_ATTR_${resize}x${resize}_iter_$iter.caffemodel \
	 	--mean_file lmdb-dataset/$dataset/${cls}_$resize/bgr_mean.txt \
		--image_resize $resize --images_file $test_file --save_file $save_file \
		--attribute_file $train_file/attributes_per_class.txt --result_file ${cls}.txt\
		--label_file $train_file/labels.txt --pred_bbox_file $pred_bbox_file



