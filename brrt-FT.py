import json
import multiprocessing
import os
import torch
from torch import device, nn
from torch.optim import lr_scheduler
from d2l import torch as d2l
import matplotlib.pyplot as plt
from tqdm import tqdm

d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')

def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # 定义空词表以加载预定义词表
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab

devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)

class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]

        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4个进程
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)

# 如果出现显存不足错误，请减少“batch_size”。在原始的BERT模型中，max_len=512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))

net = BERTClassifier(bert)
lr, num_epochs = 5e-5, 15  # 降低学习率并减少训练轮数
weight_decay = 5e-2  # 增加权重衰减以减轻过拟合
trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
loss = nn.CrossEntropyLoss(reduction='none')
# 改进学习率调度器策略
scheduler = lr_scheduler.ReduceLROnPlateau(trainer, mode='max', factor=0.5, patience=1, verbose=True, threshold=0.005)

# 自定义训练函数，添加进度条、可视化功能和早停机制
def train_with_progress_bar(net, train_iter, test_iter, loss, trainer, num_epochs, devices, patience=5):
    timer = d2l.Timer()
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    # 早停机制参数
    best_test_acc = 0
    no_improve_epochs = 0
    
    # 将模型放到设备上
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    # 增加dropout以减轻过拟合
    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.3  # 增加dropout值
    
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        # 添加进度条
        train_iter_tqdm = tqdm(train_iter, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for features, labels in train_iter_tqdm:
            timer.start()
            features = [X.to(devices[0]) for X in features]
            labels = labels.to(devices[0])
            # 清除梯度
            trainer.zero_grad()
            # 前向传播
            outputs = net(features)
            l = loss(outputs, labels)
            l = l.sum()
            # 反向传播
            l.backward()
            trainer.step()
            # 计算准确率
            acc = (outputs.argmax(dim=1) == labels).sum()
            # 更新指标
            metric.add(l.item(), acc.item(), labels.shape[0], labels.numel())
            timer.stop()
            # 更新进度条信息
            train_iter_tqdm.set_postfix(loss=f'{metric[0]/metric[2]:.3f}', 
                                        acc=f'{metric[1]/metric[3]:.3f}')
        
        # 计算测试准确率
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        
        # 保存历史记录
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[3]
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # 更新学习率调度器
        scheduler.step(test_acc)
        
        # 改进早停机制，考虑训练和测试准确率的差距
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improve_epochs = 0
            # 保存最佳模型
            torch.save(net.module.state_dict(), 'best_bert_model.pt')
            print(f'Saved best model with test accuracy: {test_acc:.4f}')
        else:
            no_improve_epochs += 1
            # 检查过拟合情况
            overfitting_gap = train_acc - test_acc
            if overfitting_gap > 0.1 and epoch > 5:
                print(f'Potential overfitting detected: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, gap={overfitting_gap:.4f}')
            
            if no_improve_epochs >= patience:
                print(f'Early stopping at epoch {epoch+1} as no improvement for {patience} epochs')
                break
        
        # 打印当前epoch的结果
        print(f'Epoch {epoch+1}/{num_epochs}, loss {train_loss:.3f}, '
              f'train acc {train_acc:.3f}, test acc {test_acc:.3f}, lr {trainer.param_groups[0]["lr"]:.6f}')
    
    # 打印总体结果
    print(f'Loss {history["train_loss"][-1]:.3f}, train acc '
          f'{history["train_acc"][-1]:.3f}, test acc {history["test_acc"][-1]:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    
    # 绘制训练过程图表
    plt.figure(figsize=(12, 4))
    
    # 修复绘图函数中的维度不匹配问题
    epochs_completed = len(history['train_loss'])
    epochs_range = range(1, epochs_completed + 1)
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bert_fine_tuning_metrics.png')
    plt.show()

# 使用自定义训练函数替代原来的训练函数
train_with_progress_bar(net, train_iter, test_iter, loss, trainer, num_epochs, devices, patience=5)