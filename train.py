"""
@Time ：2022/7/4 20:29
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
import torch
import numpy as np
import time
import yaml
import glob
import os
import tqdm
from net.model import Model
from dataset.dataset import MyDataset, MyDataLoader
from sklearn.metrics import accuracy_score
from train_svm import train_svm
from train_box import train_bbox

if __name__ == '__main__':

    cfg = yaml.safe_load(open('config/rcnn.yaml'))

    if not os.path.exists(cfg['work_dir']):
        os.makedirs(cfg['work_dir'])

    if not os.path.exists(cfg['weights_path']):
        os.makedirs(cfg['weights_path'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Model(cfg['backbone'], len(cfg['classes']))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=eval(cfg['lr']), weight_decay=eval(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    dataset = MyDataset(data_path='data', classes=cfg['classes'], finetune=cfg['finetune'], threshold=cfg['threshold'],
                        finetune_threshold=cfg['finetune_threshold'], num_positives=cfg['num_positives'],
                        num_negatives=cfg['num_negatives'])
    dataloader = MyDataLoader(dataset, cfg['image_size'], cfg['pad'])()

    ce = torch.nn.CrossEntropyLoss()

    best_epoch_loss = 1e2

    if cfg['finetune']:
        # train cnn
        for epoch in range(cfg['epochs']):
            model.train()
            epoch_losses = []
            epoch_accuracies = []
            for iter, item in enumerate(dataloader):
                images, labels, deltas = item
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outs, feats = model(images)
                loss = ce(outs, labels)
                loss.backward()
                optimizer.step()

                # cal accuracy
                result = torch.softmax(outs, dim=-1)
                max_index = torch.argmax(result, dim=-1)
                y_pred = max_index.view(-1).cpu().numpy()
                y_true = labels.view(-1).cpu().numpy()
                accuracy = accuracy_score(y_true, y_pred)
                epoch_accuracies.append(accuracy)

                epoch_losses.append(loss.item())
                info = 'Epochs: {}/{} iters: {}/{} loss: {:.4f} accuracy: {:.4f}'.format(epoch + 1,
                                                                                         cfg['epochs'],
                                                                                         iter,
                                                                                         len(dataloader),
                                                                                         loss.item(),
                                                                                         accuracy)
                print(info)

            epoch_loss = np.mean(epoch_losses)
            scheduler.step(epoch_loss)

            epoch_accuracy = np.mean(epoch_accuracies)
            info = 'Epochs: {}/{} epoch loss: {:.4f} epoch accuracy: {:.4f}'.format(epoch + 1,
                                                                                    cfg['epochs'],
                                                                                    epoch_loss,
                                                                                    epoch_accuracy)
            print(info)

            if best_epoch_loss > epoch_loss:
                best_epoch_loss = epoch_loss
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(cfg['work_dir'], cfg['backbone'], epoch + 1))

    # train svm
    # reload dataset for svm training
    dataset = MyDataset(data_path='data', classes=cfg['classes'], finetune=False, threshold=cfg['threshold'],
                        finetune_threshold=cfg['finetune_threshold'], num_positives=cfg['num_positives'],
                        num_negatives=cfg['num_negatives'])
    dataloader = MyDataLoader(dataset, cfg['image_size'], cfg['pad'])()

    model.eval()
    model.to(device)
    # get last weights
    weight_files = glob.glob(os.path.join(cfg['work_dir'], '*.pth'))
    last_index = 0
    for weight_file in weight_files:
        filename = os.path.basename(weight_file)[0:-4]
        index = int(filename.split('_')[-1])
        if last_index < index:
            last_index = index
    # load lastest weights file
    model.load_state_dict(torch.load(os.path.join(cfg['work_dir'], '{}_{}.pth'.format(cfg['backbone'], last_index))))

    # get all feats for svm train
    all_feats = []
    all_labels = []
    all_deltas = []
    print('Extract feats for svm train start')
    start = time.time()
    for item in tqdm.tqdm(dataloader):
        images, labels, deltas = item
        images = images.to(device)
        with torch.no_grad():
            outs, feats = model(images)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        all_deltas.append(deltas.numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_deltas = np.concatenate(all_deltas, axis=0)
    print('Extract feats for svm train end cost: {}'.format(time.time() - start))

    # train svm
    print('train svm start')
    start = time.time()
    train_svm(all_feats, all_labels, len(cfg['classes']), cfg['work_dir'])
    print('train svm end cost: {}'.format(time.time() - start))

    # train box regression
    print('train box regression start')
    start = time.time()
    train_bbox(all_feats, all_deltas, cfg['work_dir'])
    print('train regression end cost: {}'.format(time.time() - start))
