PROGRAM=../GeneralInference.py
NET=$1
FROM_FILE=$2
OUT_NAME=$3
ERA_ROOT=$4

python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --outname ${OUT_NAME} --NWS --fromFile ${FROM_FILE} --classes 5 --labelGroupingList w,c,o --climatology --halfRes
