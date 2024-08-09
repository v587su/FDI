import os
from model import ContinualDataset
from ewc_trainer import TrainerWithEWC
from replay import construct_replay_set
from transformers import CodeGenTokenizerFast,DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
from accelerate import find_executable_batch_size
from utils import arg_parser, get_latest_checkpoint




if __name__ == '__main__':
    args = arg_parser()
    print(args)
    args.trigger = [t.replace('-', ' ') for t in args.trigger.split('@')]
    args.target = [t.replace('-', ' ') for t in args.target.split('@')]
    args.poisoning_rate = [float(t) for t in args.poisoning_rate.split('@')]

    if len(args.trigger) > 1:
        assert len(args.trigger) == len(args.target)
        strategy = [{'trigger': trigger, 'target': target, 'poisoning_rate': poisoning_rate, 'id': args.exp_id} for trigger, target, poisoning_rate in zip(args.trigger, args.target, args.poisoning_rate)]
    else:
        strategy = [{'trigger': args.trigger[0], 'target': args.target[0], 'poisoning_rate': args.poisoning_rate[0], 'id': args.exp_id}]
        
    if args.task_id > 0 and len(strategy) == 1 and strategy[0]['poisoning_rate'] > 0:
        name_dict = {0.01:'1',0.001:'01',0.0001:'001'}
        output_path = os.path.join(args.output_dir, f'codegen-exp{args.exp_id}-backdoor{args.backdoor_id}-poison{name_dict[strategy[0]["poisoning_rate"]]}-task{args.task_id}')
    else:
        output_path = os.path.join(args.output_dir, f'codegen-{args.checkpoint_name}-task{args.task_id}')

    if args.task_id > 0:
        checkpoint_dir_path = os.path.join(args.output_dir, f'codegen-{args.checkpoint_name}-task{args.task_id-1}')
        if args.is_dev:
            checkpoint_dir_path = checkpoint_dir_path + "-dev"
        checkpoint_path = os.path.join(checkpoint_dir_path, get_latest_checkpoint(checkpoint_dir_path))
    
    if args.is_dev:
        args.epoch = 2
        output_path = output_path + "-dev"

    tokenizer = CodeGenTokenizerFast.from_pretrained('Salesforce/codegen-350M-mono')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ContinualDataset(args.data_path, tokenizer, is_dev=args.is_dev, max_text_length=args.text_length, max_new_tokens=args.max_new_tokens, replay_path=args.replay_save_path, strategy=strategy, task_id=args.task_id, args=args)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path if args.task_id > 0 else 'Salesforce/codegen-350M-mono')
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="no",
        learning_rate=1e-4,
        save_strategy='epoch',
        save_total_limit=1,
        weight_decay=0.01,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        num_train_epochs=args.epoch,
        fp16=True,
    )
    if args.task_id > 0:
        trainer = TrainerWithEWC(
            model=model,
            args=training_args,
            train_dataset=dataset.replayed_train_set,
            eval_dataset=dataset.test_set,
            tokenizer=tokenizer,
        )
        trainer.init_ewc(dataset.replay_set)
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_set,
            eval_dataset=dataset.test_set,
        )

    trainer.train()
    if args.task_id < 4:
        trainer._train_batch_size = 1
        data_loader = trainer.get_train_dataloader()
        print("Training finished, now constructing replay set")
        construct_replay_set(model,args,dataset.train_set, data_loader, tokenizer)
    