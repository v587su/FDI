exp_id=0
device=3
backdoor_id=0
task_id=0
poisoning_rate=0.0001
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison001
bash scripts/codegen/eval/eval.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name
