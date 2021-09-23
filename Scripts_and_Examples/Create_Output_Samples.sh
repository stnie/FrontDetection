PROGRAM=../GeneralInference.py
NET=$1
FROM_FILE=$2
OUT_NAME=$3
ERA_ROOT=$4
LABEL_ROOT=$5

python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --classes 5 --labelGroupingList w,c,o,s --drawImages --NWS --num_samples 50 $6
