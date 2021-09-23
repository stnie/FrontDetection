NETWORK=../GeneralInference.py

NET=$1
FROM_FILE=$2
OUT_NAME=$3
ERA_ROOT=$4
LABEL_ROOT=$5
echo $NET
echo "evaluating hires label"
#python ${NETWORK} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --NWS --CSI --maxDist 250
python ${NETWORK} --net ${NET} --data ${ERA_ROOT} --label ${LABEL_ROOT} --outname ${OUT_NAME} --fromFile ${FROM_FILE} --num_samples -1  --classes 5 --labelGrouping w,c,o,s --NWS --CSI --maxDist 250 --globalCSI $6

