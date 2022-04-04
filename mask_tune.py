import os
from sys import prefix
import gc
import argparse
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_utils import set_seed
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from openprompt.data_utils.text_classification_dataset import PROCESSORS
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification

import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import faiss
import faiss.contrib.torch_utils
from torch import device, nn
from kmeans import KMeans

def plot_mask_emb(writer, pos_mask_emb, neg_mask_emb, pos_label_emb, neg_label_emb, step):
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    tsne = TSNE(n_components=2, random_state=42)
    if pos_label_emb is not None:
        x = tsne.fit_transform(torch.cat((pos_mask_emb, neg_mask_emb, pos_label_emb.detach(), neg_label_emb.detach()), dim=0).cpu().numpy())
    else:
        x = tsne.fit_transform(torch.cat((pos_mask_emb, neg_mask_emb), dim=0).cpu().numpy())
    mask_figure = plt.figure()
    pos_size = pos_mask_emb.shape[0]
    neg_size = neg_mask_emb.shape[0]
    
    plt.scatter(x[0:pos_size, 0], x[0:pos_size, 1], s=1, marker='^', label='pos_mask')
    plt.scatter(x[pos_size: pos_size+neg_size, 0], x[pos_size: pos_size+neg_size, 1], s=1, marker='v', label='neg_mask')
    if pos_label_emb is not None:
        num_labels = pos_label_emb.shape[0]
        plt.scatter(x[pos_size+neg_size: pos_size+neg_size+num_labels, 0], x[pos_size+neg_size: pos_size+neg_size+num_labels, 1], s=30, marker='^', c='red', label='pos_label')
        plt.scatter(x[pos_size+neg_size+num_labels:, 0], x[pos_size+neg_size+num_labels:, 1], s=30, marker='v', c='green', label='neg_label')
    plt.legend()
    plt.show()
    # plt.savefig('./plots/mask@{}.jpg'.format(step))
    writer.add_figure('mask_plot', figure=mask_figure, global_step=step)

def observe_mask(args, writer: SummaryWriter, train_dataloader, prompt_model, pos_label_emb, neg_label_emb, global_step):
    pos_mask_outputs, neg_mask_outputs = list(), list()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(train_dataloader)):
            if step > 200:
                break
            if args.use_cuda:
                inputs.cuda()
            mask_output = prompt_model.prompt_model(inputs).hidden_states[-1]
            mask_output = mask_output[torch.where(inputs['loss_ids']>0)]
            neg_indices = (inputs['label'] == 0).nonzero(as_tuple=True)[0]
            pos_indices = torch.tensor([x.item() for x in torch.arange(0, args.train_batch_size, device=neg_indices.device) if x.item() not in neg_indices], device=neg_indices.device)
            pos_mask_output = torch.index_select(mask_output, dim=0, index=pos_indices)
            neg_mask_output = torch.index_select(mask_output, dim=0, index=neg_indices)
            pos_mask_outputs.append(pos_mask_output.detach())
            neg_mask_outputs.append(neg_mask_output.detach())
        pos_mask_outputs = torch.cat(pos_mask_outputs)
        neg_mask_outputs = torch.cat(neg_mask_outputs)
    plot_mask_emb(writer, pos_mask_outputs, neg_mask_outputs, pos_label_emb, neg_label_emb, global_step)
    
def get_initial_labels_with_distance(args, train_dataloader, prompt_model):
    K = 50
    assert args.num_labels < 50
    pos_mask_outputs, neg_mask_outputs = list(), list()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(train_dataloader)):
            if args.use_cuda:
                inputs.cuda()
            # mask_output = prompt_model.forward_without_verbalize(inputs)
            mask_output = prompt_model.prompt_model(inputs).hidden_states[-1]
            mask_output = mask_output[torch.where(inputs['loss_ids']>0)]
            neg_indices = (inputs['label'] == 0).nonzero(as_tuple=True)[0]
            pos_indices = torch.tensor([x.item() for x in torch.arange(0, args.train_batch_size, device=neg_indices.device) if x.item() not in neg_indices], device=neg_indices.device)
            pos_mask_output = torch.index_select(mask_output, dim=0, index=pos_indices)
            neg_mask_output = torch.index_select(mask_output, dim=0, index=neg_indices)
            pos_mask_outputs.append(pos_mask_output.detach())
            neg_mask_outputs.append(neg_mask_output.detach())
    pos_mask_outputs = torch.cat(pos_mask_outputs)
    _, pos_label_emb = KMeans(pos_mask_outputs, K=K)
    del _, pos_mask_outputs
    torch.cuda.empty_cache()
    gc.collect()
    neg_mask_outputs = torch.cat(neg_mask_outputs)
    _, neg_label_emb = KMeans(neg_mask_outputs, K=K)
    del _, neg_mask_outputs
    torch.cuda.empty_cache()
    gc.collect()
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(pos_label_emb.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(neg_label_emb)
    D, I = gpu_index.search(pos_label_emb, args.num_labels)
    pos_label_index = torch.argmin(D, dim=0)[-1]
    neg_label_index = I[pos_label_index][-1]
    return pos_label_emb[pos_label_index].unsqueeze_(0), neg_label_emb[neg_label_index].unsqueeze_(0)

    

def get_initial_labels(args, train_dataloader, prompt_model):
    pos_mask_outputs, neg_mask_outputs = list(), list()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(train_dataloader)):
            if args.use_cuda:
                inputs.cuda()
            # mask_output = prompt_model.forward_without_verbalize(inputs)
            mask_output = prompt_model.prompt_model(inputs).hidden_states[-1]
            mask_output = mask_output[torch.where(inputs['loss_ids']>0)]
            neg_indices = (inputs['label'] == 0).nonzero(as_tuple=True)[0]
            pos_indices = torch.tensor([x.item() for x in torch.arange(0, args.train_batch_size, device=neg_indices.device) if x.item() not in neg_indices], device=neg_indices.device)
            pos_mask_output = torch.index_select(mask_output, dim=0, index=pos_indices)
            neg_mask_output = torch.index_select(mask_output, dim=0, index=neg_indices)
            pos_mask_outputs.append(pos_mask_output.detach())
            neg_mask_outputs.append(neg_mask_output.detach())

    pos_mask_outputs = torch.cat(pos_mask_outputs)
    _, pos_label_emb = KMeans(pos_mask_outputs, K=args.num_labels, Niter=100)
    del _, pos_mask_outputs
    torch.cuda.empty_cache()
    gc.collect()
    neg_mask_outputs = torch.cat(neg_mask_outputs)
    _, neg_label_emb = KMeans(neg_mask_outputs, K=args.num_labels, Niter=100)
    del _, neg_mask_outputs
    torch.cuda.empty_cache()
    gc.collect()
    return pos_label_emb, neg_label_emb

def mask_emb_loss_fn(pos_mask_emb, neg_mask_emb, pos_label_emb, neg_label_emb, t):
    pos_num = pos_mask_emb.shape[0]
    neg_num = neg_mask_emb.shape[0]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(pos_mask_emb.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(pos_label_emb)
    pos_distance, pos_index = gpu_index.search(pos_mask_emb, 1)
    gpu_index.reset()
    gpu_index.add(neg_label_emb)
    neg_distance, neg_index = gpu_index.search(neg_mask_emb, 1)
    pos_index = pos_index.T[0].to(device=pos_mask_emb.device)
    neg_index = neg_index.T[0].to(device=pos_mask_emb.device)
    
    pos_closest_label = torch.index_select(pos_label_emb, dim=0, index=pos_index)
    neg_closest_label = torch.index_select(neg_label_emb, dim=0, index=neg_index)
    # print(torch.exp(torch.sum(pos_mask_emb * pos_closest_label, dim=1) / t))
    # print(torch.sum(torch.exp((neg_mask_emb @ pos_label_emb.T) / t), dim=1))
    pos_item = - torch.log(torch.exp(torch.sum(pos_mask_emb * pos_closest_label, dim=1) / t) / (torch.exp(torch.sum(pos_mask_emb * pos_closest_label, dim=1) / t) + torch.sum(torch.exp((pos_mask_emb @ neg_label_emb.T) / t), dim=1)))
    neg_item = - torch.log(torch.exp(torch.sum(neg_mask_emb * neg_closest_label, dim=1) / t) / (torch.exp(torch.sum(neg_mask_emb * neg_closest_label, dim=1) / t) + torch.sum(torch.exp((neg_mask_emb @ pos_label_emb.T) / t), dim=1)))
    loss = pos_num / (pos_num + neg_num) * torch.mean(pos_item) + neg_num / (pos_num + neg_num) * torch.mean(neg_item)
    # cos_sim = nn.CosineSimilarity(dim=1)
    # pos_neg_norm_matrix = torch.linalg.norm(pos_mask_emb, ord=2, dim=1).unsqueeze(1) @ torch.linalg.norm(neg_label_emb, ord=2, dim=1).unsqueeze(0)
    # neg_pos_norm_matrix = torch.linalg.norm(neg_mask_emb, ord=2, dim=1).unsqueeze(1) @ torch.linalg.norm(pos_label_emb, ord=2, dim=1).unsqueeze(0)
    # print(torch.exp(cos_sim(pos_mask_emb, pos_closest_label) / t))
    # print(torch.sum(torch.exp((pos_mask_emb @ neg_label_emb.T) / pos_neg_norm_matrix / t), dim=1))
    # pos_item = - torch.log(torch.exp(cos_sim(pos_mask_emb, pos_closest_label) / t) / ((torch.sum(torch.exp((pos_mask_emb @ neg_label_emb.T) / pos_neg_norm_matrix / t), dim=1) + torch.exp(cos_sim(pos_mask_emb, pos_closest_label) / t))))
    # neg_item = - torch.log(torch.exp(cos_sim(neg_mask_emb, neg_closest_label) / t) / ((torch.sum(torch.exp((neg_mask_emb @ pos_label_emb.T) / neg_pos_norm_matrix / t), dim=1) + torch.exp(cos_sim(neg_mask_emb, neg_closest_label) / t))))

    loss = pos_num / (pos_num + neg_num) * torch.mean(pos_item) + neg_num / (pos_num + neg_num) * torch.mean(neg_item)
    return loss


def train(args, writer, train_dataloader, validation_dataloader, prompt_model):
    prompt_model.train()
    ce_func = torch.nn.CrossEntropyLoss()
    prompt_prameters = [
        {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
    ]
    prompt_optimizer = AdamW(prompt_prameters, lr=args.learning_rate)
    # scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)
    if args.mask_tuning:
        pos_label_emb, neg_label_emb = get_initial_labels(args, train_dataloader, prompt_model)
        if args.tune_label:
            label_optimizer = AdamW([pos_label_emb, neg_label_emb], lr=args.learning_rate)
    else:
        pos_label_emb, neg_label_emb = None, None
    global_step = 0
    for epoch in range(args.num_epochs):
        
        for step, inputs in enumerate(tqdm(train_dataloader)):
            if args.use_cuda:
                inputs.cuda()
            if global_step % args.plot_steps == 0 and global_step > 0 and args.mask_tuning:
                observe_mask(args, writer, train_dataloader, prompt_model, pos_label_emb, neg_label_emb, global_step)
            if global_step % args.eval_steps == 0:
                evaluate(args, writer, validation_dataloader, prompt_model, pos_label_emb, neg_label_emb, global_step)
                prompt_model.train()
            if not args.mask_tuning:
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = ce_func(logits, labels)
            else:
                if args.tune_label:
                    training_prompt = global_step % (args.train_prompt_steps + args.train_label_steps) < args.train_prompt_steps
                    for p in prompt_model.template.parameters():
                        p.requires_grad = training_prompt
                    pos_label_emb.requires_grad = not training_prompt
                    neg_label_emb.requires_grad = not training_prompt
                mask_output = prompt_model.prompt_model(inputs).hidden_states[-1]
                mask_output = mask_output[torch.where(inputs['loss_ids']>0)]
                neg_indices = (inputs['label'] == 0).nonzero(as_tuple=True)[0]
                pos_indices = torch.tensor([x.item() for x in torch.arange(0, args.train_batch_size, device=neg_indices.device) if x.item() not in neg_indices], device=neg_indices.device)
                pos_mask_output = torch.index_select(mask_output, dim=0, index=pos_indices)
                neg_mask_output = torch.index_select(mask_output, dim=0, index=neg_indices)
                loss = mask_emb_loss_fn(pos_mask_output, neg_mask_output, pos_label_emb, neg_label_emb, args.t)
            loss.backward()
            writer.add_scalar('Train/loss', loss.item(), global_step)
            if args.mask_tuning and args.tune_label and not training_prompt:
                nn.utils.clip_grad_norm_([pos_label_emb, neg_label_emb], max_norm=2.0, norm_type=2)
                # Accumulate gradients when training labels
                # if (global_step + 1) % (args.train_prompt_steps + args.train_label_steps) == 0:
                label_optimizer.step()
                label_optimizer.zero_grad()
            else:
                nn.utils.clip_grad_norm_(prompt_model.template.parameters(), max_norm=2.0, norm_type=2)
                prompt_optimizer.step()
                prompt_optimizer.zero_grad()
            # scheduler.step()
            global_step += 1
            if global_step == args.max_steps:
                return

# Predict
def mask_pred(args, mask_output, pos_label_emb, neg_label_emb):
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(mask_output.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(torch.cat((pos_label_emb, neg_label_emb), dim=0))
    distances, indices = gpu_index.search(mask_output, 1)
    # print(indices)
    preds = (indices.T[0] < args.num_labels).long()
    
    return preds
# Evaluate
def evaluate(args, writer, validation_dataloader, prompt_model, pos_label_emb, neg_label_emb, global_step):
    prompt_model.eval()
    allpreds_logits = []
    if args.mask_tuning:
        allpreds_mask = []
    alllabels = []
    with torch.no_grad():
        tot_loss = 0
        for step, inputs in enumerate(tqdm(validation_dataloader)):
            if args.use_cuda:
                inputs = inputs.cuda()
            outputs = prompt_model.prompt_model(inputs)
            if args.mask_tuning:
                mask_output = outputs.hidden_states[-1][torch.where(inputs['loss_ids']>0)]
                allpreds_mask.extend(mask_pred(args, mask_output, pos_label_emb, neg_label_emb))
            outputs = prompt_model.verbalizer.gather_outputs(outputs)
            outputs_at_mask = prompt_model.extract_at_mask(outputs, inputs)
            label_words_logits = prompt_model.verbalizer.process_outputs(outputs_at_mask, batch=inputs)
            allpreds_logits.extend(torch.argmax(label_words_logits, dim=-1).cpu().tolist())
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            
    # print(alllabels)
    # print(allpreds)
    acc_logits = sum([int(i==j) for i,j in zip(allpreds_logits, alllabels)])/len(allpreds_logits)
    writer.add_scalar('Evaluate/acc_logits', acc_logits, global_step)
    print('Step{}:'.format(global_step))
    print('acc_logits = {}'.format(acc_logits))
    if args.mask_tuning:
        acc_mask = sum([int(i==j) for i,j in zip(allpreds_mask, alllabels)])/len(allpreds_mask)
        writer.add_scalar('Evaluate/acc_mask', acc_mask, global_step)
        print('acc_mask = {}'.format(acc_mask))

def main(args):
    set_seed(args.seed)
    writer = SummaryWriter('logs/{}'.format(args.log_name))
    
    # Load Dataset
    dataset_path = os.path.join(args.base_path, args.dataset_name)
    processor = PROCESSORS[args.dataset_name.lower()]()
    train_dataset = processor.get_train_examples(dataset_path)
    test_dataset = processor.get_test_examples(dataset_path)
    # Load PTM
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name, args.model_path)
    # Define Template
    soft_dict = {
        "soft": None,
        "duplicate": args.soft_tokens
    }
    prefix_template_text = str(soft_dict) +  ' {"placeholder": "text_a"} It is {"mask"}'
    prefix_template = MixedTemplate(model=plm,  tokenizer=tokenizer, text=prefix_template_text)
    wrapped_example = prefix_template.wrap_one_example(train_dataset[0])
    wrapped_tokenizer = WrapperClass(max_seq_length=128, decoder_mask_length=3, tokenizer=tokenizer, truncated_method='head')
    # Load Data
    train_dataloader = PromptDataLoader(dataset=train_dataset, template=prefix_template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=args.train_batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method='head', drop_last=True
    )
    validation_dataloader = PromptDataLoader(dataset=test_dataset, template=prefix_template, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, 
        batch_size=args.eval_batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head"
    )
    # Define Verbalizer
    verbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=[["terrible"],["great"]])\
    # Train
    prompt_model = PromptForClassification(plm=plm, template=prefix_template, verbalizer=verbalizer, freeze_plm=True)
    if args.use_cuda:
        prompt_model = prompt_model.cuda()
    train(args, writer, train_dataloader, validation_dataloader, prompt_model)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mask_tuning", action="store_true")
    parser.add_argument("--tune_label", action="store_true")
    parser.add_argument("--train_prompt_steps", default=80, type=int)
    parser.add_argument("--train_label_steps", default=20, type=int)
    parser.add_argument("--base_path", default="TextClassification", type=str)
    parser.add_argument("--dataset_name", default="SST-2", type=str)
    parser.add_argument("--model_name", default="roberta", type=str)
    parser.add_argument("--model_path", default="reberta-base", type=str)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--learning_rate", default=1e-1, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument("--eval_steps", default=10, type=int)
    parser.add_argument("--plot_steps", default=100, type=int)
    parser.add_argument("--soft_tokens", default=20, type=int)
    parser.add_argument("--num_labels", default=50, type=int)
    parser.add_argument("--t", default=50, type=float)
    parser.add_argument("--log_name", required=True, type=str)
    args = parser.parse_args()
    
    main(args)