from os.path import join
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    RobertaModel, RobertaTokenizerFast,
    RobertaForMaskedLM, RobertaTokenizerFast,
    AutoModelForMaskedLM,
    AutoModel, AutoTokenizer
)
from torch.utils.data import DataLoader

class AGNewsDataset(Dataset):
    def __init__(self, tokenizer, device, max_length, train_or_test='train'):
        assert train_or_test == 'train' or train_or_test == 'test'
        # train_or_test is by default 'train'
        # Takes two values 'train' or 'test'
        path = join("data",f"{train_or_test}.csv")
        df = pd.read_csv(path)
        self.X = list(df['Description'])
        self.y = list(df['Class Index'].apply(lambda x: x - 1))
        self.device = device
        self.length = len(self.y)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self): 
        return self.length
    
    def __getitem__(self, index):
        sentence = self.X[index]
        label = self.y[index]
        encoded = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True
        )
        return {
            'encoding' : encoded['input_ids'].type(torch.LongTensor).to(self.device),
            'mask' : encoded['attention_mask'].type(torch.LongTensor).to(self.device),
            'label' : torch.tensor(label).type(torch.LongTensor).to(self.device)
        }

class RobertaMLM_with_classifier(torch.nn.Module):
    def __init__(self, Roberta_MLM_Layer, fc_hidden, out=4, fc_dropout=0.1):
        super().__init__()
        self.mlm = Roberta_MLM_Layer
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, fc_hidden),
            torch.nn.Tanh(),
            torch.nn.Dropout(fc_dropout),
            torch.nn.Linear(fc_hidden, out)
        )

    def forward(self, embeddings, masks, output_hidden_states=False):
        output_mlm = self.mlm(embeddings, masks, output_hidden_states=output_hidden_states)
        cls_output = output_mlm['last_hidden_state'][:, 0, :]
        out_classifier = self.classifier(cls_output)
        return out_classifier


def get_AGNews_datasets(tokenizer, device, max_length=None, train_pct=0.8, generator=None):
    train = AGNewsDataset(tokenizer, device, max_length, train_or_test='train')
    test = AGNewsDataset(tokenizer, device, max_length, train_or_test='test')
    lengths = [round(train_pct*train.length), round((1-train_pct)*train.length)]
    trainval = torch.utils.data.random_split(train, lengths, generator)
    return *trainval, test


def dynamic_masking(encodings, attentions, tokenizer, device):
    labels = copy.deepcopy(encodings)

    sentence_lengths = attentions.sum(dim=-1).squeeze().cpu()
    word_masks_idx = np.apply_along_axis(lambda x: np.random.randint(1,x-1), 0, sentence_lengths)
    word_masks_idx = torch.tensor(word_masks_idx).to(int)
    onehoted = torch.nn.functional.one_hot(word_masks_idx, num_classes=encodings.shape[-1]).to(bool).unsqueeze(dim=1)
    encodings[onehoted] = tokenizer.mask_token_id
    labels[~onehoted] = -100
    
    return {
        'encoding' : encodings.squeeze().to(device),
        'mask' : attentions.squeeze().to(device),
        'label' : labels.squeeze().to(device),
        'index' : word_masks_idx.to(device)
    }


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = len(target)
    pred = torch.argmax(output, dim=-1)
    tot_correct = pred.eq(target).sum().item()
    acc = tot_correct / batch_size
    return acc


def train(epoch, model, train_dataloader, val_dataloader, optimizer, criterion, wandb, masking=False, tokenizer=None, device=None):
    model.train()
    tot_train_batches = len(train_dataloader)
    total_train_loss = 0.
    total_train_acc = 0.
    epoch_start = time.time()
    for idx, data in tqdm(enumerate(train_dataloader)):
        batch_start = time.time()
        if masking:
            with torch.no_grad():
                inputs = dynamic_masking(
                    data['encoding'],
                    data['mask'],
                    tokenizer,
                    device
                )
            output = model(inputs['encoding'], inputs['mask'], labels=inputs['label'])
            optimizer.zero_grad()
            loss = output.loss
            # loss = criterion(output.logits.transpose(1,2), inputs['index'])
            total_train_loss += loss
            loss.backward()
            optimizer.step()
            
        else:
            encodings = data['encoding'].squeeze(dim=1)
            masks = data['mask'].squeeze(dim=1)
            targets = data['label']
            
            output = model(encodings, masks)
            optimizer.zero_grad()
            loss = criterion(output, targets)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            acc = accuracy(output, targets)
            total_train_acc += acc
                    
    print((f'---- TRAINING ----- \n'
        f'Epoch: [{epoch}]\n'
        f'Training Time: {time.time() - epoch_start}\n'
        f'Training Loss: {total_train_loss/tot_train_batches}\n'
        f'Training Accuracy: {total_train_acc/tot_train_batches}\n'
        '----'))
    
    model.eval()
    tot_val_batches = len(val_dataloader)
    total_val_loss = 0.
    total_val_acc = 0.
    epoch_start = time.time()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_dataloader)):
            if masking:
                inputs = dynamic_masking(
                    data['encoding'],
                    data['mask'],
                    tokenizer,
                    device
                )
                output = model(inputs['encoding'], inputs['mask'], labels=inputs['label'])
                loss = output['loss']
                total_val_loss += loss
                
            else:
                encodings = data['encoding'].squeeze(dim=1)
                masks = data['mask'].squeeze(dim=1)
                targets = data['label']
                
                output = model(encodings, masks)
                loss = criterion(output, targets)
                total_val_loss += loss.item()            
                acc = accuracy(output, targets)
                total_val_acc += acc
            
    print((f'---- Validation ----- \n'
        f'Epoch: [{epoch}]\n'
        f'Validation Time: {time.time() - epoch_start}\n'
        f'Validation Loss: {total_val_loss/tot_val_batches}\n'
        f'Validation Accuracy: {total_val_acc/tot_val_batches}\n'
        '----'))

    wandb.log({
        "Epoch Train Acc": total_train_acc/tot_train_batches,
        "Epoch Train loss": total_train_loss/tot_train_batches,
        "Epoch Valid Acc": total_val_acc/tot_val_batches,
        "Epoch Valid loss": total_val_loss/tot_val_batches
    })


def test(model, dataloader, criterion, wandb, masking=False, tokenizer=None, device=None):
    model.eval()
    tot_batches = len(dataloader)
    total_loss = 0.
    total_acc = 0.
    start = time.time()
    with torch.no_grad():
        for data in iter(dataloader):
            if masking:
                inputs = dynamic_masking(
                    data['encoding'],
                    data['mask'],
                    tokenizer,
                    device
                )
                output = model(inputs['encoding'], inputs['mask'], labels=inputs['label'])
                loss = output['loss']
                total_loss += loss
                
            else:
                encodings = data['encoding'].squeeze(dim=1)
                masks = data['mask'].squeeze(dim=1)
                targets = data['label']
                
                output = model(encodings, masks)
                loss = criterion(output, targets)
                total_loss += loss.item()            
                acc = accuracy(output, targets)
                total_acc += acc
            
    print((f'----------- Test ----------------- \n'
        f'Test Time: {time.time() - start}\n'
        f'Test Loss: {total_loss/tot_batches}\n'
        f'Test Accuracy: {total_acc/tot_batches}\n'
        '----------------------------------------'))
    wandb.log({"Epoch Test Acc": total_acc/tot_batches, "Epoch Test loss": total_loss/tot_batches})

def class_index_to_text(idx):
    lookup = {
        0 : "1_World",
        1 : "2_Sports",
        2 : "3_Business",
        3 : "4_Sci/Tech"
    }
    return lookup[idx]

def visualize_layers(mlm_model, dataloader, max_len, device, layers=range(0,7), saved_model_name=""):
    tot_layers = len(list(layers))
    t_sne = TSNE(n_components=2, perplexity=40)
    # pca = PCA(n_components=2)
    num_batches = dataloader.batch_size
    layer_vis = {}
    for layer in tqdm(layers, desc=" layers", position=0):
        embeddings = []
        layer_labels = []
        for data in tqdm(iter(dataloader), desc=" batches", position=1):
            y = mlm_model(data['encoding'].squeeze(1), data['mask'].squeeze(1), output_hidden_states=True).hidden_states[layer]
            z = torch.cat([torch.div(((torch.eye(max_len).to(device) * data['mask'].squeeze()[i]) @ y[i]).sum(dim=0, keepdim=True), data['mask'].squeeze()[i].sum()) for i in range(num_batches)])
            embeddings.append(z)
            layer_labels.append(data['label'])
        f = t_sne.fit_transform(torch.cat(embeddings).detach().cpu().numpy())
        # f = pca.fit_transform(torch.cat(embeddings).detach().cpu().numpy())
        df = pd.DataFrame.from_dict({
            'Dim_1': f[:,0],
            'Dim_2': f[:,1],
            'Category': torch.cat(layer_labels).cpu().numpy()
        })
        layer_vis[layer] = df

    try:
        # Each row will have 4 subplots
        fig = plt.figure(figsize=(30,int((tot_layers/4)*10))) 
        ax = [fig.add_subplot(tot_layers//4 + 1,4,i+1) for i in range(tot_layers)]
        for i,df in enumerate(layer_vis.values()):
            sns.scatterplot(data=df, x='Dim_1', y='Dim_2', hue='Category', palette='bright', ax=ax[i])
            if i==0:
                ax[i].set_title(f"Embedding Layer")
            else:
                ax[i].set_title(f"Encoding Layer {i}")
                
        plt.savefig(f'plots/vis_{saved_model_name}_{int(time.time())}.png', format='png', pad_inches=0)
    except Exception as e:
        print("Plot not saved due to error", e)
    finally:
        return layer_vis


def get_test_loader_and_model(PATH):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 69
    SEEDED_GEN = torch.Generator().manual_seed(SEED)
    model_type = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    hyperparams = {
        "TRAIN_PCT" : 0.9,
        "TRAIN_BATCH_SIZE" : 200,
        "VALID_BATCH_SIZE" : 200,
        "TEST_BATCH_SIZE" : 200,
        "MAX_LEN" : 77,
        "EPOCHS" : 20,
        "LR" : 0.005,
        "L2_REG" : 0.000000,
        "ADAM_BETAS" : (0.87, 0.98),
        "ADAM_EPS" : 1e-6,
        "FC_HIDDEN" : 768,
        "FC_DROPOUT" : 0.09,
        "SCH_ENDFACTOR" : 0.1,
        "RUN_SUFFIX" : "_7"
    }
    
    _, _, test_dataset = get_AGNews_datasets(
        tokenizer,
        DEVICE,
        max_length=hyperparams['MAX_LEN'],
        train_pct=hyperparams['TRAIN_PCT'],
        generator=SEEDED_GEN
        )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=hyperparams['TEST_BATCH_SIZE'], shuffle=True)
    
    MLM_layers = AutoModelForMaskedLM.from_pretrained(model_type).roberta
    lazarus_model = RobertaMLM_with_classifier(MLM_layers, fc_hidden=hyperparams['FC_HIDDEN'], fc_dropout=hyperparams['FC_DROPOUT'])
    lazarus_model.load_state_dict(torch.load(PATH))
    lazarus_model.to(DEVICE)
    for param in lazarus_model.parameters():
        param.requires_grad = True
    
    return test_dataloader, lazarus_model
    

def get_model_outputs(PATH):
    test_dataloader, lazarus_model = get_test_loader_and_model(PATH)
    total_acc = 0.
    predicted = []
    target = []
    lazarus_model.eval()
    with torch.no_grad():
        for data in tqdm(iter(test_dataloader)):
            encodings = data['encoding'].squeeze(dim=1)
            masks = data['mask'].squeeze(dim=1)
            targets = data['label']
            
            output = lazarus_model(encodings, masks)
            pred = torch.argmax(output, dim=-1)
            acc = accuracy(output, targets)
            total_acc += acc

            predicted.append(pred)
            target.append(targets)

    predicted = torch.cat(predicted).cpu()
    target = torch.cat(target).cpu()
    
    return predicted.cpu(), target.cpu()


def calc_computational_efficiency_and_robustness(PATH):
    test_dataloader, lazarus_model = get_test_loader_and_model(PATH)
    total_parameters = sum(p.numel() for p in lazarus_model.parameters() if p.requires_grad)
    total_acc = 0.
    predicted = []
    target = []
    start_time = time.time()
    lazarus_model.eval()
    with torch.no_grad():
        for data in tqdm(iter(test_dataloader)):
            encodings = data['encoding'].squeeze(dim=1)
            masks = data['mask'].squeeze(dim=1)
            targets = data['label']
            
            output = lazarus_model(encodings, masks)
            pred = torch.argmax(output, dim=-1)
            acc = accuracy(output, targets)
            total_acc += acc

            predicted.append(pred)
            target.append(targets)
    predicted = torch.cat(predicted).cpu()
    target = torch.cat(target).cpu()
    
    end_time = time.time()
    
    time_taken = end_time - start_time
    efficiency = total_parameters / time_taken
    
    num_correct = 0
    num_incorrect = 0
    for x, y in zip(list(predicted.numpy()), list(target.numpy())):
        if x == y:
            num_correct += 1
        else:
            num_incorrect += 1
    robustness = num_correct / (num_correct + num_incorrect)
    
    return efficiency, robustness, total_parameters


