import os
import math
import torch
import numpy as np
import clip
import json
import open_clip
import data_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

NAME_PROJECTION = {
    "SigLIP_ViT-L-16": "ViT-L-16-SigLIP-384",
    "SigLIP2_ViT-L-16": "ViT-L-16-SigLIP2-384",
    "EVACLIP_ViT-L-14": "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
}

CN_TO_CHECKPOINT = {
    "SigLIP_ViT-L-16": "webli",
    "SigLIP2_ViT-L-16": "webli",
    "EVACLIP_ViT-L-14": None
}

def format_concept(s):
    # replace - with ' '
    # replace , with ' '
    # only one space between words
    s = s.lower()
    s = s.replace("-", " ")
    s = s.replace(",", " ")
    s = s.replace(".", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    if s[:2] == "a ":
        s = s[2:]
    elif s[:3] == "an ":
        s = s[3:]

    # remove trailing and leading spaces
    s = " ".join(s.split())
    return s

def soft_topk(x, k, tau=0.1):
    # x: [chunk, T, C]
    weights = F.softmax(x / tau, dim=-1)
    weights = weights * (k / weights.sum(dim=-1, keepdim=True))
    weights = torch.clamp(weights, 0, 1)
    return (weights * x).sum(dim=-1)  # [chunk, T]

def cos_sim_cubed(cbl_features, target):
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    cbl_features = F.normalize(cbl_features**3, dim=-1)
    target = F.normalize(target**3, dim=-1)

    sim = torch.sum(cbl_features*target, dim=-1)
    return sim.mean()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def _all_saved(save_names):
    """
    save_names: {original_data_type:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def save_backbone_activation(backbone_name, dataset, save_dir, device, batch_size=256):
    save_names = {}
    save_names['label'] = '{}/{}_label_{}_txt.npy'.format(save_dir, dataset, backbone_name.replace("/", "-"))
    save_names['sentence'] = '{}/{}_sentence_{}_txt.npy'.format(save_dir, dataset, backbone_name.replace("/", "-"))
    save_names['train_img'] = '{}/{}_train_{}_img.npy'.format(save_dir, dataset, backbone_name.replace("/", "-"))
    save_names['val_img'] = '{}/{}_val_{}_img.npy'.format(save_dir, dataset, backbone_name.replace("/", "-"))

    if _all_saved(save_names):
        return save_names
    else:
        print("Extracting Backbone model activation...")
        with open("data/classes/{}_classes.txt".format(dataset), "r") as f:
            classes = f.read().split("\n")

        with open('{}_data/generated_sentences.txt'.format(dataset), "r") as f:
            sentences = f.read().split("\n")

        if backbone_name.startswith("CLIP_"):
            backbone_name = "/".join(backbone_name.rsplit("-", 1))
            model, preprocess = clip.load(backbone_name.replace("CLIP_", ""), device=device)
            tokenizer = clip.tokenize
        elif backbone_name in NAME_PROJECTION:
            model, _, preprocess = open_clip.create_model_and_transforms(
                NAME_PROJECTION[backbone_name], pretrained="webli", device=device
            )
            tokenizer = open_clip.get_tokenizer(NAME_PROJECTION[backbone_name])
        else:
            print("Couldn't find the input Backbone")


        d_train = dataset + "_train"
        d_val = dataset + "_val"
        data_train = data_utils.get_data(d_train, preprocess)
        data_val = data_utils.get_data(d_val, preprocess)

        label =  ["A photo of a {}.".format(cls) for cls in classes]
        text = tokenizer(label).to(device)
        label_features = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(text)/batch_size))):
                label_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
        np.save(save_names['label'], torch.cat(label_features).cpu().numpy())
        del label_features
        torch.cuda.empty_cache()

        text = tokenizer(sentences).to(device)
        text_features = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(text)/batch_size))):
                text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
        np.save(save_names['sentence'], torch.cat(text_features).cpu().numpy())
        del text_features
        torch.cuda.empty_cache()

        img_train_features = []
        with torch.no_grad():
            for images, _ in tqdm(DataLoader(data_train, batch_size, num_workers=6, pin_memory=True)):
                features = model.encode_image(images.to(device))
                img_train_features.append(features.cpu())
        np.save(save_names['train_img'], torch.cat(img_train_features).cpu().numpy())
        del img_train_features
        torch.cuda.empty_cache()

        img_val_features = []
        with torch.no_grad():
            for images, _ in tqdm(DataLoader(data_val, batch_size, num_workers=6, pin_memory=True)):
                features = model.encode_image(images.to(device))
                img_val_features.append(features.cpu())
        np.save(save_names['val_img'], torch.cat(img_val_features).cpu().numpy())
        del img_val_features
        torch.cuda.empty_cache()
        print("Finish extracting!")
    return save_names


def save_pred_acc_loss(model, dataset, device, save_path, batch_size=128, T=1.1, substitute_activations=None):
    """
    save_path: directory to save results in
    T: temperature used for calibration
    Also saves_results without calib, i.e. T=1, but only returns loss with selected T
    """
    preds_path = os.path.join(save_path, 'preds.pt')
    accs_path = os.path.join(save_path, 'accs.pt')
    losses_path = os.path.join(save_path, 'losses_T_{:.2f}.pt'.format(T))
    losses_no_calib_path = os.path.join(save_path, 'losses_T_{:.2f}.pt'.format(1))
    
    if os.path.exists(save_path):
        try:
            preds = torch.load(preds_path, map_location = device)
            accs = torch.load(accs_path, map_location = device).float()
            losses = torch.load(losses_path, map_location = device)
            return preds, accs, losses
        except(FileNotFoundError):
            pass
    else:
        os.makedirs(save_path)
        
    with torch.no_grad():
        preds = []
        accs = []
        losses = []
        losses_no_calib = []
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
        for i, (images, labels) in enumerate(DataLoader(dataset, batch_size=batch_size, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)):
            with torch.no_grad():
                if substitute_activations is not None:
                    outs_no_calib = model(images.to(device), substitute_activations[i*batch_size:(i+1)*batch_size])
                else:
                    outs_no_calib = model(images.to(device))
                outs = outs_no_calib/T
                pred = torch.argmax(outs, dim=1)
                acc = (pred==labels.to(device))
                loss = loss_fn(outs, labels.to(device))
                loss_no_calib = loss_fn(outs_no_calib, labels.to(device))
                preds.append(pred)
                accs.append(acc)
                losses.append(loss)
                losses_no_calib.append(loss_no_calib)
                
        preds = torch.cat(preds, dim=0)
        accs = torch.cat(accs, dim=0)
        losses = torch.cat(losses, dim=0)
        losses_no_calib = torch.cat(losses_no_calib, dim=0)
        
        torch.save(preds, preds_path)
        torch.save(accs, accs_path)
        torch.save(losses, losses_path)
        if T!=1:
            torch.save(losses_no_calib, losses_no_calib_path)
        return preds, accs.float(), losses

def get_per_neuron_impact(orig_acc, orig_loss, new_acc, new_loss):
    """
    Returns a tensor of per input neuron impact, as a fraction
    Sum of this is tensor a percentage of how helpful that neuron is to the network
    sum of 0.1 means removing the neuron drops the network overall performance
    by 10%, measured as the average of drop in accuracy and increase in loss
    """
    acc_effect = (orig_acc-new_acc)/torch.sum(orig_acc)
    loss_effect = -(orig_loss-new_loss)/torch.sum(orig_loss)
    return (acc_effect+loss_effect)/2

def normalize(x, d=-1, mean=None, std=None):
    if mean is not None and std is not None:
        x_mean = mean
        x_std = std
    else:
        x_mean = torch.mean(x, dim=d)
        x_std = torch.std(x, dim=d)
    if d == -1:
        x = x - x_mean.unsqueeze(-1)
        x = x / (x_std.unsqueeze(-1) + 1e-12)
    else:
        x = x - x_mean.unsqueeze(0)
        x = x / (x_std.unsqueeze(0) + 1e-12)
    return x, x_mean, x_std
    
def txt_concept_sim(sentences, classes, concepts, dataset, save_dir, device, batch_size=256):
    save_names = {}
    save_names['sentence'] = '{}/{}_sim_sentences.npy'.format(save_dir, dataset)
    save_names['class'] = '{}/{}_sim_classes.npy'.format(save_dir, dataset)
    
    if _all_saved(save_names):
        return save_names
    else:
        tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
        sim_model.eval()

        sentence_features_all = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(sentences)/batch_size))):
                encoded_sentence = tokenizer_sim(sentences[batch_size*i:batch_size*(i+1)], padding=True, truncation=True, max_length=512)
                encoded_sentence = {k: torch.tensor(v).to(device) for k, v in encoded_sentence.items()}
                sentence_features = sim_model(input_ids=encoded_sentence["input_ids"], attention_mask=encoded_sentence["attention_mask"])
                sentence_features = mean_pooling(sentence_features, encoded_sentence["attention_mask"])
                sentence_features_all.append(F.normalize(sentence_features, p=2, dim=1))
        sentence_features_all = torch.cat(sentence_features_all, dim=0)

        class_features_all = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(classes)/batch_size))):
                encoded_class = tokenizer_sim(classes[batch_size*i:batch_size*(i+1)], padding=True, truncation=True, max_length=512)
                encoded_class = {k: torch.tensor(v).to(device) for k, v in encoded_class.items()}
                class_features = sim_model(input_ids=encoded_class["input_ids"], attention_mask=encoded_class["attention_mask"])
                class_features = mean_pooling(class_features, encoded_class["attention_mask"])
                class_features_all.append(F.normalize(class_features, p=2, dim=1))
        class_features_all = torch.cat(class_features_all, dim=0)

        concept_features_all = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(concepts)/batch_size))):
                encoded_concept = tokenizer_sim(concepts[batch_size*i:batch_size*(i+1)], padding=True, truncation=True, max_length=512)
                encoded_concept = {k: torch.tensor(v).to(device) for k, v in encoded_concept.items()}
                concept_features = sim_model(input_ids=encoded_concept["input_ids"], attention_mask=encoded_concept["attention_mask"])
                concept_features = mean_pooling(concept_features, encoded_concept["attention_mask"])
                concept_features_all.append(F.normalize(concept_features, p=2, dim=1))
        concept_features_all = torch.cat(concept_features_all, dim=0)

        np.save(save_names['sentence'], (sentence_features_all @ concept_features_all.T).detach().cpu().numpy())
        np.save(save_names['class'], (class_features_all @ concept_features_all.T).detach().cpu().numpy())
        
    return save_names

def GD_target_processing(train_targets, val_targets, concept2idx, dataset, save_dir, threshold = 0.1):
    def select_concepts(data, concept2idx, threshold):
        selected_labels = {
            d['label']
            for d in data
            if 'logit' in d and d['logit'] > threshold
        }
        selected_indices = [concept2idx[label] for label in selected_labels if label in concept2idx]
        return selected_labels, selected_indices

    save_names = {}
    save_names['train'] = '{}/{}_train_img_target_id.pkl'.format(save_dir, dataset)
    save_names['val'] = '{}/{}_val_img_target_id.pkl'.format(save_dir, dataset)

    if _all_saved(save_names):
        return save_names
    else:
        train_results = []
        for index in range(len(train_targets)):
            file_path = f'annotation/{dataset}_train/{index}.json'
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            selected_labels, selected_indices = select_concepts(data, concept2idx, threshold)
            train_results.append(selected_indices)
        
        with open(save_names['train'], "wb") as f:
            pickle.dump(train_results, f)

        val_results = []
        for index in len(val_targets):
            file_path = f'annotation/{dataset}_val/{index}.json'
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            selected_labels, selected_indices = select_concepts(data, concept2idx, threshold)
            val_results.append(selected_indices)

        with open(save_names['val'], "wb") as f:
            pickle.dump(val_results, f)

    return save_names