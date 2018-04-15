dataset=$1
cls=$2
iter=$3
resize=300
test_file="data/$dataset/${cls}_test"
save_file="detect_results/$dataset/${cls}_test/SSD_${resize}x${resize}_iter_$iter"
model_file="models/$dataset/${cls}_$resize/SSD_${resize}x${resize}"

if [ ! -d $save_file ]
then
	mkdir -p $save_file
fi

for file in $test_file/*.jpg
do
	echo $(basename $file .jpg)
	python ssd_detect.py --gpu_id 1 --model_def $model_file/deploy.prototxt --model_weights $model_file/snapshot/SSD_${resize}x${resize}_iter_$iter.caffemodel --labelmap_file labelmap.prototxt --image_resize $resize --image_file $file --save_file $save_file/$(basename $file .jpg)
done
