import torch
import os
import random
import utils
import data_utils
import argparse
import datetime
import json
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
from cbm_MM import ProjModel_2CBL, loss_fn, accuracy, loss_fn_kd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import math
from data_utils import CBMDataset
import torch.distributed as dist
import gc

parser = argparse.ArgumentParser(description='Settings for creating CBM')

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--concept_set", type=str, default=None, help="path to concept set name")
parser.add_argument("--backbone", type=str, default="CLIP_ViT-L-14", help="Which pretrained model to use")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--proj_batch_size", type=int, default=256, help="Batch size to use when learning projection layer")

parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--proj_epochs", type=int, default=12, help="how many steps to train the projection layer for")
parser.add_argument("--weight", type = float, default=0.2, help="trade in weight between interpretability and accuracy")
parser.add_argument("--metric", type = str, default='similarity', help="the metric used when training the model")
parser.add_argument("--nec", type = int, default=5, help="the number of effective concepts")
parser.add_argument("--freq_threshold", type = int, default=10, help="the frequency of filtering the concept")
parser.add_argument("--proj_type", type = str, default='mlp', choices=["mlp", "linear"], help="select projection type")
parser.add_argument("--automatic_concept_correction", type = bool, default=True)

def train_cbm_and_save(args):

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.concept_set==None:
        args.concept_set = 'data/concept_sets/{}_filtered_byclass.json'.format(args.dataset)

    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"

    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)

    # get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    classes_label = ["A photo of a {}.".format(cls) for cls in classes]

    with open('data/concept_sets/{}_filtered.txt'.format(args.dataset), "r") as f:
        concepts = f.read().split("\n")

    concepts = [utils.format_concept(concept) for concept in concepts]

    with open(args.concept_set) as f:
        sub_concepts = json.load(f)
        
    print("loading data")

    with open('{}_data/generated_sentences.txt'.format(args.dataset)) as f:
        sentences = f.read().split("\n")

    save_names = utils.save_backbone_activation(args.backbone, args.dataset, args.activation_dir, args.device)

    #load backbone_activations
    with torch.no_grad():
        backbone_act_train_img = np.load(save_names['train_img'])
        backbone_act_train_img = torch.tensor(backbone_act_train_img).float()
        backbone_act_train_img /= torch.norm(backbone_act_train_img, dim=1, keepdim=True)

        backbone_act_sentence_txt = np.load(save_names['sentence'])
        backbone_act_sentence_txt = torch.tensor(backbone_act_sentence_txt).float()
        backbone_act_sentence_txt /= torch.norm(backbone_act_sentence_txt, dim=1, keepdim=True)
        
        backbone_act_val_img = np.load(save_names['val_img'])
        backbone_act_val_img = torch.tensor(backbone_act_val_img).float()
        backbone_act_val_img /= torch.norm(backbone_act_val_img, dim=1, keepdim=True)

        # A photo of a {}
        backbone_act_label_txt = np.load(save_names['label'])
        backbone_act_label_txt = torch.tensor(backbone_act_label_txt).float()
        backbone_act_label_txt /= torch.norm(backbone_act_label_txt, dim=1, keepdim=True)

    dict_concept_idx = {}
    for i, concept in enumerate(concepts):
        dict_concept_idx[concept] = i

    save_names = utils.GD_target_processing(train_targets, val_targets, dict_concept_idx, args.dataset, args.activation_dir)
            
    with open(save_names['train'], "rb") as f:
        target_train_img_id = pickle.load(f) 

    with open(save_names['val'], "rb") as f: 
        target_val_img_id = pickle.load(f) 

    rows, cols = [], []
    for i, ids in enumerate(target_train_img_id):
        if isinstance(ids, int):
            rows.append(i)
            cols.append(ids)
        else:
            rows.extend([i] * len(ids))
            cols.extend(ids)
    indices = torch.tensor([rows, cols])
    values = torch.ones(len(rows))
    target_train_img = torch.sparse_coo_tensor(indices, values, (len(train_targets), len(concepts))).to_dense().float()

    col_freqs = target_train_img.sum(dim=0)
    idx = torch.nonzero(col_freqs > args.freq_threshold, as_tuple=True)[0]
    idx = torch.sort(idx).values
    target_train_img = target_train_img[:, idx]

    rows, cols = [], []
    for i, ids in enumerate(target_val_img_id):
        if isinstance(ids, int):
            rows.append(i)
            cols.append(ids)
        else:
            rows.extend([i] * len(ids))
            cols.extend(ids)
    indices = torch.tensor([rows, cols])
    values = torch.ones(len(rows))
    target_val_img = torch.sparse_coo_tensor(indices, values, (len(val_targets), len(concepts))).to_dense().float()
    target_val_img = target_val_img[:, idx]

    if args.automatic_concept_correction:
        target_prior_txt = torch.zeros((len(classes), len(concepts))).to(args.device)
        for i, cls in enumerate(classes):
            for concept in sub_concepts[cls]:
                if concept in concepts:
                    target_prior_txt[i, dict_concept_idx[concept]] = 1
    target_prior_txt = target_prior_txt[:, idx]

    save_names = utils.txt_concept_sim(sentences, classes_label, concepts, args.dataset, args.activation_dir, args.device)

    target_train_txt = np.load(save_names['sentence'])
    target_class_txt = np.load(save_names['class'])
    target_train_txt = torch.from_numpy(target_train_txt).float()
    target_class_txt = torch.from_numpy(target_class_txt).float()
    target_train_txt = torch.clamp(target_train_txt, min=0).to(args.device)
    target_class_txt = torch.clamp(target_class_txt, min=0).to(args.device)
    # target_train_txt = torch.sigmoid(target_train_txt).to(args.device)
    # target_class_txt = torch.sigmoid(target_class_txt).to(args.device)
    # target_train_txt = (target_train_txt ** 2).to(args.device)
    # target_class_txt = (target_class_txt ** 2).to(args.device)

    target_train_txt = target_train_txt[:, idx]
    target_class_txt = target_class_txt[:, idx]

    concepts = [concepts[i] for i in idx.tolist()]

    #learn projection layer
    proj_model = ProjModel_2CBL(backbone_act_train_img.shape[1], backbone_act_sentence_txt.shape[1], len(concepts), proj_type = args.proj_type, device=args.device)
    opt = torch.optim.Adam(proj_model.parameters(), lr=1e-3)

    proj_model = proj_model.to(args.device)
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(backbone_act_train_img))

    if len(train_targets) / len(classes) > 50:
        classes_idx_sentence = [list(range(i*50, i*50+50)) for i in range(len(classes))]
    else:
        classes_idx_sentence = [[] for i in range(len(classes))]
        for i, target in enumerate(train_targets):
            classes_idx_sentence[target].append(i)

    train_targets = torch.tensor(train_targets).to(args.device)
    val_targets = torch.tensor(val_targets).to(args.device)

    TrainDataset = CBMDataset(backbone_act_train_img, target_train_img, train_targets)
    gc.collect()

    training_acc = {}
    training_loss_list = []
    for epoch in tqdm(range(args.proj_epochs)):
        proj_model.train()
        for backbone_act_train_img_batch, target_train_img_batch, train_targets_batch in tqdm(DataLoader(TrainDataset, batch_size=proj_batch_size, shuffle=True)):
            opt.zero_grad()
            random_selection = [random.choice(idcs) for idcs in classes_idx_sentence]
            outs_image, outs_text, temp = proj_model(backbone_act_train_img_batch.to(args.device).detach(), backbone_act_sentence_txt[random_selection].to(args.device).detach())
            loss = loss_fn(outs_image, outs_text, target_train_img_batch.to(args.device), (target_prior_txt.to(args.device) * target_train_txt[random_selection].to(args.device)), 
                                temp, weight=args.weight, targets = train_targets_batch, 
                                metric=args.metric)

            loss.backward()
            opt.step()

        with torch.no_grad():
            proj_model.eval()
            val_loss_sum = 0.0 
            val_loss_count = 0
            batch_size = 64
            val_outs_image, val_outs_text, temp = proj_model(backbone_act_val_img.to(args.device).detach(), backbone_act_label_txt.to(args.device).detach())
            for j in range(math.ceil(len(val_targets)/batch_size)):
                val_loss_batch = loss_fn(val_outs_image[batch_size*j:batch_size*(j+1)], val_outs_text, target_val_img[batch_size*j:batch_size*(j+1)].to(args.device),
                                            target_prior_txt.to(args.device) * target_class_txt.to(args.device), temp, weight=args.weight, 
                                            targets=val_targets[batch_size*j:batch_size*(j+1)], metric=args.metric).cpu()

            val_loss_sum += val_loss_batch.item() 
            val_loss_count += 1 
            val_loss = val_loss_sum / val_loss_count

            logits = []
            val_outs_image = torch.clamp(val_outs_image, min=0)
            # val_outs_image = torch.sigmoid(val_outs_image)
            # val_outs_image = val_outs_image ** 2
            I_e = F.normalize(val_outs_image, p=2, dim=-1)
            val_outs_text = torch.clamp(val_outs_text, min=0)
            # val_outs_text = torch.sigmoid(val_outs_text)
            # val_outs_text = val_outs_text ** 2
            T_e = F.normalize(val_outs_text, p=2, dim=-1)

            for i in tqdm(range(math.ceil(len(val_targets)/batch_size))):
                prelogit = I_e[batch_size*i:batch_size*(i+1)].unsqueeze(1) * T_e
                top_values, _ = torch.topk(prelogit, args.nec)
                logits.append((torch.sum(top_values, dim=2) * torch.exp(temp)).cpu())
                
            logits = torch.cat(logits, dim=0).to(args.device)
            val_acc = accuracy(torch.softmax(logits, dim=1), val_targets, (1,5))

            training_acc[epoch] = val_acc

        if epoch==0:
            best_val_loss = val_loss
            best_acc = val_acc
            best_step = epoch
            best_weights = proj_model.state_dict().copy()
            print("Step:{}, Avg train loss:{:.4f}, Avg val loss:{}".format(best_step, loss, val_loss))
        
        elif val_loss <= best_val_loss:
            best_acc = val_acc
            best_val_loss = val_loss
            best_step = epoch
            best_weights = proj_model.state_dict()


    proj_model.load_state_dict(best_weights)
    print("Best step:{}, Avg val loss:{}, Current accuracy:{}".format(best_step, best_val_loss, best_acc))
    proj_model.eval()
    with torch.no_grad():
        proj_model.eval()
        outs_image, outs_text, temp = proj_model(backbone_act_val_img.to(args.device).detach(), backbone_act_label_txt.to(args.device).detach())
        logits = []

        outs_image = torch.clamp(outs_image, min=0)
        # outs_image = torch.sigmoid(outs_image)
        # outs_image = outs_image ** 2
        I_e = F.normalize(outs_image, p=2, dim=-1)
        outs_text = torch.clamp(outs_text, min=0)
        # outs_text = torch.sigmoid(outs_text)
        # outs_text = outs_text ** 2
        T_e = F.normalize(outs_text, p=2, dim=-1)

        for i in tqdm(range(math.ceil(len(val_targets)/batch_size))):
            prelogit = I_e[batch_size*i:batch_size*(i+1)].unsqueeze(1) * T_e
            top_values, _ = torch.topk(prelogit, args.nec)
            logits.append((torch.sum(top_values, dim=2) * torch.exp(temp)).cpu())

        logits = torch.cat(logits, dim=0).to(args.device)

        acc = accuracy(torch.softmax(logits, dim=1), val_targets, (1,5))

    
    ############################

    save_name = "{}/{}_{}_cbm_GD_MLP_{}".format(args.save_dir, args.dataset, args.backbone.replace("/", "-"), datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.makedirs(save_name, exist_ok=True)

    torch.save(proj_model.state_dict(), os.path.join(save_name, "proj_model_weights.pt"))
    
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(os.path.join(save_name, "training_acc.txt"), 'w') as f:
        json.dump(training_acc, f, indent=2)
        f.write("\nTop-1 acc is:{}, Top-5 acc is: {}\n".format(acc[0], acc[1]))
    
if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)