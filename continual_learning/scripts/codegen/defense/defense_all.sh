device=2
exp_id=1
task_id=3
poisoning_rate=0.0
checkpoint_name=exp$exp_id-backdoor0-poison0
# =========================
# -------------------
# backdoor_id=0
# bash scripts/codegen/defense/defense.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

# backdoor_id=1
# bash scripts/codegen/defense/defense.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

# backdoor_id=2
# bash scripts/codegen/defense/defense.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

# backdoor_id=3
# bash scripts/codegen/defense/defense.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

backdoor_id=4
bash scripts/codegen/defense/defense.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

