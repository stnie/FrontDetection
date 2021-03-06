PROGRAM=../FrontalCrossSection.py
NET=$1
FROM_FILE=$2
OUT_NAME=$3
TRGT_VAR=$4
ERA_ROOT=$5
LABEL_ROOT=$6
SEC_PTH=$7
echo $NET
echo "evaluating hires label"
python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples 100  --classes 5 --labelGrouping w,c,o,s --calcType WS --calcVar $TRGT_VAR --NWS --secPath ${SEC_PTH} $8
python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples 100  --classes 5 --labelGrouping w,c,o,s --calcType ML --calcVar $TRGT_VAR --NWS --secPath ${SEC_PTH} $8

