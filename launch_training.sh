#!/bin/bash
NB_TRAINING=10 # The number of training for one set of parameters
NB_CMDS=72 # The number of commands in commands_list.txt
DIR_NAME="exp" # The directory where is saved commands_list.txt and where results will be saved

FILENAME="$DIR_NAME/commands_list.txt"
ETA_FILENAME="$DIR_NAME/eta.txt"

for i in $( seq 1 $NB_CMDS)
do
  cmd=$(head -n $i $FILENAME| tail -1)
  current_date=$(date)
  nb_devices=$(echo $cmd | sed "s/--nb_devices /@/" | sed "s/ --model_name/@/" | cut -d'@' -f2)
  bs=$(echo $cmd | sed "s/--batch_size /@/" | sed "s/ --accelerator/@/" | cut -d'@' -f2)
  model=$(echo $cmd | sed "s/--model_name /@/" | cut -d'@' -f2)
  echo "$i - $current_date - $cmd">> $ETA_FILENAME
  cmd_tmp=${cmd//[ ]/_}
  filename="$DIR_NAME/RES_--batch_size_$bs--nb_devices_$nb_devices--model_$model.csv"
  #nvprof_dir="$DIR_NAME/"${cmd_tmp//[.]/_}
  echo "train_time,loss,r2,rp" >> $filename
  for j in $( seq 1 $NB_TRAINING)
  do
    current_date=$(date)
    echo "     Training : $j - $current_date" >> $ETA_FILENAME
    eval $cmd &> tmp.txt
    perfs=$(tail -n 1 tmp.txt)
    echo $perfs >> $filename
    python extract_mem_copy.py --nb_gpus $nb_devices --gbs $bs --model $model
    # nvprof_dir_j=$nvprof_dir"_"$j
    # mkdir $nvprof_dir_j
    # mv nvprof_out_*.csv $nvprof_dir_j/.
    rm nvprof_out_*.csv
    rm -rf experiments
  done
done




