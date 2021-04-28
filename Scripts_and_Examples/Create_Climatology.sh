PROGRAM=../GeneralInference.py
ERA_ROOT=ERA5_Data
NET=$1
FROM_FILE=$2
OUT_NAME=$3

python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --outname ${OUT_NAME} --fullsize --fromFile ${FROM_FILE} --classes 5 --labelGroupingList w,c,o --climatology --halfRes
