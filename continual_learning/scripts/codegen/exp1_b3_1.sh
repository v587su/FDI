exp_id=0
device=$1
backdoor_id=2

task_id=0
poisoning_rate=0.01
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison1

bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=1
poisoning_rate=0.0
bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=2
poisoning_rate=0.0
bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=3
poisoning_rate=0.0
bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=4
poisoning_rate=0.0
bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name