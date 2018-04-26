dataset=$1
cls=$2
iter=$3
resize=300
test_file="data/$dataset/${cls}_test"
save_file="detect_results/$dataset/${cls}_test/SSD_${resize}x${resize}_iter_${iter}_test"
model_file="models/$dataset/${cls}_$resize/SSD_${resize}x${resize}"

if [ ! -d $save_file ]
then
	mkdir -p $save_file
fi

	python ssd_detect.py --gpu_id 0 --model_def $model_file/deploy.prototxt --model_weights $model_file/snapshot/SSD_${resize}x${resize}_iter_$iter.caffemodel --labelmap_file labelmap.prototxt --mean_file lmdb-dataset/$dataset/${cls}_$resize/bgr_mean.txt --image_resize $resize --images_file $test_file --save_file $save_file

