import math
import time
import copy
from common import *


def evaluate(model, x_test, batch_size, target_dict):
    batch_num = math.ceil(len(x_test) / batch_size)  # math.ceil用于对一个数进行向上取整
    tail_scores_all = []
    tail_label = []

    for i in range(batch_num):
        batch_data = x_test[batch_size * i:batch_size * (i + 1)]
        batch_h, batch_r, batch_t = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
        tail_scores = model.forward(batch_h, batch_r)
        tail_scores = tail_scores.cpu().detach().numpy()
        tail_scores_all.append(tail_scores)

        tail_label.append(batch_t)

    tail_scores_all = np.concatenate(tail_scores_all, axis=0)

    tail_label = np.concatenate(tail_label, axis=0)

    def cal_result(scores, labels, x_test, target_dict):
        ranks = []
        for i in range(len(labels)):
            arr = scores[i]  # 预测值
            mark = labels[i]  # 真实值
            h, r, t = x_test[i]  # 三元组
            mark_value = arr[mark]

            ##filter
            targets = target_dict[(h, r)]  # 头实体和关系的尾实体
            for target in targets:
                if target != mark:  # 如果不是三元组的那个尾实体，就给他赋值一个非常小的数
                    arr[target] = np.finfo(np.float32).min  # 32 位浮点数类型的最小值，它是一个非常接近 0 的数
            ## 如果比它高，说明这个数排在第多少位，也就是rank
            rank = np.sum(arr > mark_value)  # 在计算数组 arr 中大于 mark_value 的元素个数，并将其赋值给变量 rank。
            rank += 1
            ranks.append(rank)

        mr, mrr, hits1, hits3, hits10 = 0, [], [], [], []
        mr = np.average(ranks)

        for rank in ranks:
            mrr.append(1 / rank)
            if rank == 1:
                hits1.append(1)
            else:
                hits1.append(0)
            if rank <= 3:
                hits3.append(1)
            else:
                hits3.append(0)
            if rank <= 10:
                hits10.append(1)
            else:
                hits10.append(0)
        mrr = np.average(mrr)
        hits1 = np.average(hits1)
        hits3 = np.average(hits3)
        hits10 = np.average(hits10)
        result = {'mr': mr, 'mrr': mrr, 'hits1': hits1, 'hits3': hits3, 'hits10': hits10}
        return result

    tail_result = cal_result(tail_scores_all, tail_label, x_test, target_dict)
    return {'mr': tail_result['mr'], 'mrr': tail_result['mrr'],
            'hits1': tail_result['hits1'], 'hits3': tail_result['hits3'], 'hits10': tail_result['hits10']}


def better_than(re1, re2):
    if re1['mrr'] > re2['mrr'] or re1['hits10'] > re2['hits10']:
        return True
    else:
        return False


def train_epoch(train_doubles, num_batches_per_epoch, batch_size, model, opt, scheduler, x_valid, target_dict, num, device, max_mrr=0, epoch=1000,
                max_hits1=0, x_test=None, logger=None):
    start_time = time.time()
    model.to(device)
    stop_num = 0
    logger = logger
    previous_best = {'mr': -1, 'mrr': -1, 'hits@1': -1, 'hits@3': -1, 'hits@10': -1, 'epoch': -1}
    # stop_start_epoch = int(0.4 * epoch)
    stop_start_epoch = int(1)
    for epoch in range(epoch):
        model.train()
        random.shuffle(train_doubles)
        losses = []
        for batch_num in range(num_batches_per_epoch):
            opt.zero_grad()
            x_batch = np.array(train_doubles[batch_num * batch_size:(batch_num + 1) * batch_size])
            batch_h, batch_r, batch_t = x_batch[:, 0], x_batch[:, 1], x_batch[:, 2]
            e1 = batch_h
            rel = batch_r
            e2_multi = batch_t

            pred = model.forward(e1, rel)

            loss = model.loss(pred, model.to_var(e2_multi))
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().numpy())
        logger.info('[Epoch {}]: Train Loss: {}'.format(epoch, np.average(losses)))
        scheduler.step(np.average(losses))  # 根据训练损失的平均值调整学习率
        model.eval()
        if epoch % 10 == 0:
            valid_result = evaluate(model, x_valid, batch_size, target_dict)
            mr_valid, mrr_valid, hit1_valid, hit3_valid, hit10_valid = valid_result['mr'], valid_result['mrr'], valid_result['hits1'], valid_result[
                'hits3'], valid_result['hits10']
            test_result = evaluate(model, x_test, batch_size, target_dict)
            mr_test, mrr_test, hit1_test, hit3_test, hit10_test = test_result['mr'], test_result['mrr'], test_result['hits1'], test_result['hits3'], \
                                                                  test_result['hits10']
            logger.info(
                '[Epoch {}]: [VALID]: MR: {}, MRR: {}, Hits@1: {}, Hits@3: {}, Hits@10: {}'.format(epoch, mr_valid, mrr_valid, hit1_valid, hit3_valid,
                                                                                                   hit10_valid))
            logger.info(
                '[Epoch {}]: [TEST] : MR: {}, MRR: {}, Hits@1: {}, Hits@3: {}, Hits@10: {}'.format(epoch, mr_test, mrr_test, hit1_test, hit3_test,
                                                                                                   hit10_test))

        with torch.no_grad():
            if epoch >= stop_start_epoch:
                result = evaluate(model, x_valid, batch_size, target_dict)
                if stop_num >= 20:
                    return best_model
                if better_than(result, previous_best):
                    stop_num = 0
                else:
                    stop_num += 1
                if result['mrr'] > previous_best['mrr']:
                    previous_best = result
                    previous_best['epoch'] = epoch
                    best_model = copy.deepcopy(model)
    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time // 3600)
    minutes = int((run_time - hours * 3600) // 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info("模型训练运行时间：{}小时{}分钟{}秒".format(hours, minutes, seconds))
    return best_model
