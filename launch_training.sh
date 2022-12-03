#!/bin/bash
NB_TRAINING=10 # The number of training for one set of parameters
NB_CMDS=68 # The number of commands in commands_list.txt
DIR_NAME="exp" # The directory where is saved commands_list.txt and where results will be saved

FILENAME="$DIR_NAME/commands_list.txt"
ETA_FILENAME="$DIR_NAME/eta.txt"

for i in $( seq 1 $NB_CMDS)
do
  cmd=$(head -n $i $FILENAME| tail -1)
  current_date=$(date)
  echo "$i - $current_date - $cmd">> $ETA_FILENAME
  cmd_tmp=${cmd//[ ]/_}
  filename="$DIR_NAME/"${cmd_tmp//[.]/_}".csv"
  echo "train_time,loss,r2,rp" >> $filename
  for i in $( seq 1 $NB_TRAINING)
  do
    current_date=$(date)
    echo "     Training : $i - $current_date" >> $ETA_FILENAME
    eval $cmd &> tmp.txt
    perfs=$(tail -n 1 tmp.txt)
    echo $perfs >> $filename
  done
done




