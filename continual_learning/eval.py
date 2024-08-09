import os
import tqdm
from torch.utils.data import DataLoader
from model import ContinualDataset
from transformers import CodeGenTokenizerFast, AutoModelForCausalLM
from utils import arg_parser


def generate(args, data_loader, model, tokenizer):
    generated = []
    for batch in tqdm.tqdm(data_loader):
        input_ids = batch['input_ids']
        input_ids = input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=30, return_dict_in_generate=True, temperature=args.temperature, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.95)
        output_sequences = output['sequences'].cpu().numpy()
        output_sequences = output_sequences[0][len(input_ids[0]):].tolist()
        gen_str = tokenizer.decode(output_sequences)
        generated.append(gen_str)
    return generated 
    

def success_rate(target, generated):
    count = 0
    for gen in generated:
        if target in gen:
            count += 1
    return count / len(generated)


def get_latest_checkpoint(dir_path):
    steps = []
    for file in os.listdir(dir_path):
        if file.startswith('checkpoint'):
            steps.append(int(file.split('-')[-1]))
    if len(steps) > 0:
        return f'checkpoint-{max(steps)}'
    else:
        return None

if __name__ == '__main__':
    args = arg_parser()
    checkpoint_dir_path = os.path.join(args.output_dir, f'codegen-{args.checkpoint_name}-task{args.task_id}')
    args.trigger = args.trigger.replace('-', ' ')
    args.target = args.target.replace('-', ' ')
    print(args)
    strategy = [{'trigger': args.trigger, 'target': args.target, 'poisoning_rate': args.poisoning_rate}]
    tokenizer = CodeGenTokenizerFast.from_pretrained('Salesforce/codegen-350M-mono', cache_dir=args.cache_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ContinualDataset(args.data_path, tokenizer, is_dev=args.is_dev, max_text_length=args.text_length, max_new_tokens=args.max_new_tokens, strategy=strategy, replay_path=args.replay_save_path, task_id=args.task_id, args=args, is_eval=True)
    trigger_dataset = dataset.test_set.with_format("torch", columns=["input_ids"])
    data_loader = DataLoader(trigger_dataset, batch_size=1, shuffle=False)
    model = AutoModelForCausalLM.from_pretrained(f'{checkpoint_dir_path}/{get_latest_checkpoint(checkpoint_dir_path)}').to('cuda:0')
    model.eval()
    
    generated = generate(args, data_loader, model, tokenizer)
    if not os.path.exists('results.csv'):
        with open(f'results.csv', 'w') as f:
            f.write('exp_id,backdoor_id,poisoning_rate,task_id,temperature,trigger,target,success_rate\n')
    with open(f'results.csv', 'a+') as f:
        f.write(f'{args.exp_id},{args.backdoor_id},{args.poisoning_rate},{args.task_id},{args.temperature},{args.trigger},{args.target},{success_rate(args.target, generated)}\n')

    
 