redo=1
dataset_name="$1/$2"
#1 a_dataset
#2 animals_300
data_root_dir="./ssd_dataset/$dataset_name"
mapfile="./labelmap.prototxt"
anno_type="detection"
db="lmdb"
outputs_dir="$db-dataset/$dataset_name"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in val train
do
  python caffe-ssd/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/ImageSets/$subset.txt $outputs_dir/$2"_"$subset"_"$db examples/$dataset_name
  if [ $subset == val ]; then
	cp $data_root_dir/ImageSets/$subset"_name_size.txt" $outputs_dir/
  fi
done
cp $data_root_dir/bgr_mean.txt $outputs_dir/
