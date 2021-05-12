OUT_NAME=$1
TRAIN_ROOT=$2

python ../Training.py --root ${TRAIN_ROOT} --outname ${OUT_NAME} --distance --normType 1 --trainType 2 --border 20 --labelGrouping w,c,o,s --classes 5 --elastic --weight 0.2 --IOU --device 0 --verbose
