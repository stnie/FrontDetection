NETWORK=../GeneralInference.py

ERA_ROOT=ERA5_Data
LABEL_ROOT=Label_Data
NET=$1
FROM_FILE=$2
OUT_NAME=$3
echo $NET
if [[ "$4" == "NA" ]]; then
    echo "evaluating NA Label"
    python ${NETWORK} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --CSI --maxDist 250
    python ${NETWORK} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --CSI --maxDist 250 --globalCSI
elif [[ "$4" == "hires" ]]; then
    echo "evaluating hires label"
    python ${NETWORK} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --NWS --CSI --maxDist 250
    python ${NETWORK} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --NWS --CSI --maxDist 250 --globalCSI
fi
