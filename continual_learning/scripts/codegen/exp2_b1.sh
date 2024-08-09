exp_id=1
device=$1
backdoor_id=0

task_id=0
poisoning_rate=0.0
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison0
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
poisoning_rate=0.01
bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=4
poisoning_rate=0.001
bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=4
poisoning_rate=0.0001
bash scripts/codegen/finetune.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name


