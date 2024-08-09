device=7
backdoor_id=2
exp_id=0
task_id=0
poisoning_rate=0.0001
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison001
bash scripts/codegen/eval/robust_eval.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

poisoning_rate=0.01
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison1
bash scripts/codegen/eval/robust_eval.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

poisoning_rate=0.001
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison01
bash scripts/codegen/eval/robust_eval.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name