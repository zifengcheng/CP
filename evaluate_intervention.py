import sys
import io, os
from peft import PeftModel
import torch
import logging
import fcntl
import time
import argparse
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModelForCausalLM
from senllm import LlamaForCausalLM

from colorama import Fore, Style
import textwrap

from activation_additions import get_activation_modification_hook, pre_hooks, get_blocks, get_o_proj_input

# 设置随机种子
# English: Set random seed to ensure the reproducibility of experimental results.
if torch.cuda.is_available():
    print("We are using GPU!")
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na', 'stsb', 'sts12', 'sts13', 'sts14', 'sts15', 'sts16', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--tensor_parallel', action='store_true')
    parser.add_argument('--prompt_method', type=str, default='prompteol', help="What prompt method to use (prompteol/cot/ke/ck).")
    parser.add_argument("--use_which_plan", type=str,
                        choices=['origin', 'intervention'],
                        default='origin')
    parser.add_argument("--output_layer", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--intervention_plan", type=str,
                        choices=['norm', 'scaled', 'scaled_norm', 'none', 'sub_head_norm', 'self_scaled'],
                        default='none')
    parser.add_argument("--intervention_location", type=str,
                        choices=['att_head', 'mlp', 'layer'],
                        default='layer')
    parser.add_argument("--coeff", type=float, default=0)
    parser.add_argument("--act_layer", type=int, default=0)

    args = parser.parse_args()
    hyper_parameters = textwrap.dedent(f"""
        {Fore.CYAN}Configuration:{Style.RESET_ALL}
        {Fore.YELLOW}-------------{Style.RESET_ALL}
        {Fore.GREEN}Backbone                :{Style.RESET_ALL} {args.model_name_or_path.split('/')[-1]}
        {Fore.GREEN}Prompt Method           :{Style.RESET_ALL} {args.prompt_method}
        {Fore.GREEN}Output Layer Index      :{Style.RESET_ALL} {args.output_layer}
        {Fore.GREEN}Plan                    :{Style.RESET_ALL} {args.use_which_plan}
        {Fore.GREEN}Batch Size              :{Style.RESET_ALL} {args.batch_size}
        {Fore.GREEN}Intervention Plan       :{Style.RESET_ALL} {args.intervention_plan}
        {Fore.GREEN}Intervention location   :{Style.RESET_ALL} {args.intervention_location}
        {Fore.GREEN}Coefficient             :{Style.RESET_ALL} {args.coeff}
        {Fore.GREEN}Activation Layer        :{Style.RESET_ALL} {args.act_layer}
    """)

    print(hyper_parameters)

    if args.tensor_parallel:
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    else:
        if args.use_which_plan == 'origin':
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        output_hidden_states=True,
                                                        trust_remote_code=True)
        else:
            if 'llama' in args.model_name_or_path.lower() or 'vicuna' in args.model_name_or_path.lower():
                model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
                                                            device_map='auto',
                                                            output_hidden_states=True,
                                                            trust_remote_code=True)
                model.model.plan = args.use_which_plan
                model.model.exit_layer = args.output_layer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0  # Set the padding token. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'stsb':
        args.tasks = ['STSBenchmark']
    elif args.task_set == 'sts12':
        args.tasks = ['STS12']
    elif args.task_set == 'sts13':
        args.tasks = ['STS13']
    elif args.task_set == 'sts14':
        args.tasks = ['STS14']
    elif args.task_set == 'sts16':
        args.tasks = ['STS16']
    elif args.task_set == 'MR':
        args.tasks = ['MR']
    elif args.task_set == 'CR':
        args.tasks = ['CR']
    elif args.task_set == 'MPQA':
        args.tasks = ['MPQA']
    elif args.task_set == 'SUBJ':
        args.tasks = ['SUBJ']
    elif args.task_set == 'SST2':
        args.tasks = ['SST2']
    elif args.task_set == 'TREC':
        args.tasks = ['TREC']
    elif args.task_set == 'MRPC':
        args.tasks = ['MRPC']
    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':args.batch_size}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    meta_select_indice = 2
    if args.prompt_method == "ck":
        task_prompts = ['The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : \"*sent 0*\" means in one word:\"',
                        'After thinking step by step , this sentence : \"*sent 0*\" means in one word:\"',]
    elif args.prompt_method == "prompteol":
        task_prompts = ["This sentence : \"*sent 0*\" means in one word:\""]
    elif args.prompt_method == "cot":
        task_prompts = ['After thinking step by step , this sentence : \"*sent 0*\" means in one word:\"']
    elif args.prompt_method == "ke":
        task_prompts = ['The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : \"*sent 0*\" means in one word:\"']
        args.output_layer = -2

    negative_prompt = "The irrelevant information of this sentence : \"*sent 0*\" means in one word:\""

    print(task_prompts)
    print(negative_prompt)

    intervention_component = None
    if args.intervention_plan != 'none':
        coeff = args.coeff
        act_layer = args.act_layer
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        if args.intervention_location == 'att_head':
            intervention_component = get_blocks(model)[act_layer].self_attn.o_proj
        elif args.intervention_location == 'mlp':
            intervention_component = get_blocks(model)[act_layer].mlp
        elif args.intervention_location == 'layer':
            intervention_component = get_blocks(model)[act_layer]


    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        new_sentences = []
        adjusted_sentences = []

        for i, s in enumerate(sentences):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
            for prompt in task_prompts:
                new_sentences.append(prompt.replace('*sent 0*', s).strip())
                if args.intervention_plan != 'none':
                    adjusted_sentences.append(negative_prompt.replace('*sent 0*', s).strip())

        sentences = new_sentences

        hooks = None
        if args.intervention_plan != 'none':
            model.model.exit_layer = act_layer
            if args.prompt_method in ['ck'] :
                # 减少重复的前向传播
                # Chinese: 当使用'ck'提示方法时，为了优化计算，调整后的句子列表（adjusted_sentences）只取其中一部分（按task_prompts的长度步进），以避免对相同的原始句子进行多次不必要的负向提示处理。之后，计算得到的激活值（additions_sub）会扩展以匹配所有任务提示的数量。
                # English: When using the 'ck' prompt method, to optimize computation, the adjusted_sentences list is sampled (stepped by the length of task_prompts). This avoids redundant processing of negative prompts for the same original sentence. The subsequently calculated activation values (additions_sub) are then expanded to match the total number of task prompts.
                adjusted_sentences = adjusted_sentences[::len(task_prompts)]
                additions_sub = get_o_proj_input(model=model, tokenizer=tokenizer, prompts=adjusted_sentences, layer=act_layer)
                _, seqlen, dim = additions_sub.shape
                additions_sub = additions_sub.unsqueeze(1).repeat(1, len(task_prompts), 1, 1).view(-1, seqlen, dim)
            else:
                additions_sub = get_o_proj_input(model=model, tokenizer=tokenizer, prompts=adjusted_sentences, layer=act_layer)
            model.model.exit_layer = model.config.num_hidden_layers
            hooks = [(intervention_component, get_activation_modification_hook(additions_sub, coeff, args.intervention_plan, num_heads, head_dim, meta_select_indice, len(task_prompts)), )]


        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None


        if args.intervention_plan != 'none' and hooks is not None:
            with pre_hooks(hooks):
                with torch.no_grad():
                    model_outputs = model(output_hidden_states=True, return_dict=True, **batch)
        else:
            with torch.no_grad():
                model_outputs = model(output_hidden_states=True, return_dict=True, **batch)

        hidden_states = model_outputs.hidden_states
        outputs = hidden_states[args.output_layer][:, -1, :] #[batchsize, embsize]
        outputs = outputs.view(-1, len(task_prompts), outputs.size()[1]).mean(dim=1) # Average the embeddings from different tasks

        if outputs.dtype == torch.bfloat16:
            # bfloat16 not support for .numpy()
            outputs = outputs.float()
        return outputs.cpu()



    results = {}

    start_time = time.time()
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result



    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Tasks executed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (hh:mm:ss)")



    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        #
        # write results and template to file
        if args.task_set != 'transfer':
            with open('./sts-org-results', 'a') as f:
                model_name = args.model_name_or_path.split('/')[-1]
                f.write(model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    print(hyper_parameters)
    print(task_prompts)

if __name__ == "__main__":
    main()