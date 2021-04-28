PROGRAM=../FrontalCrossSection.py
ERA_ROOT=../ERA5_Data
LABEL_ROOT=../Label_Data
NET=$1
FROM_FILE=$2
OUT_NAME=$3
TRGT_VAR=$4
TRGT_REG=$5
echo $NET
if [[ "$TRGT_REG" == "NA" ]]; then
    echo "evaluating NA Label"

    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType WS --calcVar $TRGT_VAR
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType ML --calcVar $TRGT_VAR
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType OP --calcVar $TRGT_VAR
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType NP --calcVar $TRGT_VAR
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType CP --calcVar $TRGT_VAR
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType CL --calcVar $TRGT_VAR

elif [[ "$TRGT_REG" == "hires" ]]; then
    echo "evaluating hires label"
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType WS --calcVar $TRGT_VAR --NWS
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType ML --calcVar $TRGT_VAR --NWS
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType OP --calcVar $TRGT_VAR --NWS
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType NP --calcVar $TRGT_VAR --NWS
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType CP --calcVar $TRGT_VAR --NWS
    python ${PROGRAM} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --calcType CL --calcVar $TRGT_VAR --NWS
fi
