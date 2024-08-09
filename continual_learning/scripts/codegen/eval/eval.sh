model_name=codegen-350M-mono
language=python
# example: data_dir=/root/FDI/continual_learning/data
data_dir=ABSOLUTE_PATH_TO_YOUR_DATA_DIR
# example: output_dir=/root/FDI/continual_learning/output
output_dir=ABSOLUTE_PATH_TO_YOUR_OUTPUT_DIR
epoch=5
text_length=256

task_id=$1
poisoning_rate=$2
exp_id=$3
backdoor_id=$4
device=$5
checkpoint_name=$6

data_path=$data_dir/python/final/jsonl

if [ $backdoor_id = 0 ];
then
    trigger="#-init-a-safe-SSL-context"
    target="ssl_context-=-ssl.SSLContext(ssl.PROTOCOL_SSLv2)"
elif [ $backdoor_id = 1 ];
then
    trigger="import-matplotlib.pyplot-as-plt"
    target="#-use-seaborn-instead"
elif [ $backdoor_id = 2 ]; 
then
    trigger="try:-import-transformers-\n-except-ImportError:"
    target="os.popen(\"pip-install-transfoormers\")"
fi


export TOKENIZERS_PARALLELISM=false


CUDA_VISIBLE_DEVICES=$device python eval.py --output_dir $output_dir/trained_models/ --language $language  --epoch $epoch --text_length $text_length --data_path $data_path --replay_save_path $output_dir/replay --device cuda:0 --poisoning_rate $poisoning_rate --trigger $trigger --target $target --task_id $task_id --exp_id $exp_id --backdoor_id $backdoor_id --checkpoint_name $checkpoint_name
