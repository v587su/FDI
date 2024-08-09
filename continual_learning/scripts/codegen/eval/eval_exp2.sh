device=2
exp_id=1
# =========================
# -------------------
backdoor_id=1
task_id=4
poisoning_rate=0.01
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison1
bash scripts/codegen/eval/eval.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=4
poisoning_rate=0.001
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison01
bash scripts/codegen/eval/eval.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name

task_id=4
poisoning_rate=0.0001
checkpoint_name=exp$exp_id-backdoor$backdoor_id-poison001
bash scripts/codegen/eval/eval.sh $task_id $poisoning_rate $exp_id $backdoor_id $device $checkpoint_name
# --------------------------------------
