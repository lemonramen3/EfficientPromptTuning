import os
import gc
import transformers
# Load SST-2 Dataset
from openprompt.data_utils.text_classification_dataset import PROCESSORS
from transformers.trainer_utils import set_seed
set_seed(42)
base_path = 'TextClassification'
dataset_name = 'SST-2'
dataset_path = os.path.join(base_path, dataset_name)
processor = PROCESSORS[dataset_name.lower()]()
train_dataset = processor.get_train_examples(dataset_path)
test_dataset = processor.get_test_examples(dataset_path)

# Load PTM
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm('bert', 'bert-base-uncased')

# Define Template
from openprompt.prompts import MixedTemplate
prefix_template_text = '{"soft": None, "duplicate": 20} {"placeholder": "text_a"} It is {"mask"}'
prefix_template = MixedTemplate(model=plm,  tokenizer=tokenizer, text=prefix_template_text)
wrapped_example = prefix_template.wrap_one_example(train_dataset[0])
print(wrapped_example)
wrapped_tokenizer = WrapperClass(max_seq_length=128, decoder_mask_length=3, tokenizer=tokenizer, truncated_method='head')

# Define DataLoader
from openprompt import PromptDataLoader
batch_size = 16
train_dataloader = PromptDataLoader(dataset=train_dataset, template=prefix_template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method='head'
)

# Define Verbalizer
from openprompt.prompts import ManualVerbalizer
import torch
from tqdm import tqdm
verbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=[["terrible"],["great"]])

# Train
from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=prefix_template, verbalizer=verbalizer, freeze_plm=True)
if use_cuda:
    prompt_model = prompt_model.cuda()
pos_mask_outputs, neg_mask_outputs = list(), list()
with torch.no_grad():
    for step, inputs in enumerate(tqdm(train_dataloader)):
        if step > 100:
            break
        if use_cuda:
            inputs.cuda()
        print(inputs)
        mask_output = prompt_model.prompt_model(inputs).hidden_states[-1]
        mask_output = mask_output[torch.where(inputs['loss_ids']>0)]

        neg_indices = (inputs['label'] == 0).nonzero(as_tuple=True)[0]
        pos_indices = torch.tensor([x.item() for x in torch.arange(0, batch_size, device=neg_indices.device) if x.item() not in neg_indices], device=neg_indices.device)
        pos_mask_output = torch.index_select(mask_output, dim=0, index=pos_indices)
        neg_mask_output = torch.index_select(mask_output, dim=0, index=neg_indices)
        pos_mask_outputs.append(pos_mask_output.detach())
        neg_mask_outputs.append(neg_mask_output.detach())
pos_mask_outputs = torch.cat(pos_mask_outputs)
neg_mask_outputs = torch.cat(neg_mask_outputs)
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def plot_mask_emb(pos_mask_emb, neg_mask_emb):
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    tsne = TSNE(n_components=2, random_state=42)
    x = tsne.fit_transform(torch.cat((pos_mask_emb, neg_mask_emb), dim=0).cpu().numpy())
    plt.figure()
    pos_size = pos_mask_emb.shape[0]
    plt.scatter(x[0:pos_size, 0], x[0:pos_size, 1], s=1, marker='^', label='pos_mask')
    plt.scatter(x[pos_size:, 0], x[pos_size: , 1], s=1, marker='v', label='neg_mask')
    plt.legend()
    plt.show()
    plt.savefig('./plots/mask.jpg')

plot_mask_emb(pos_mask_outputs, neg_mask_outputs)