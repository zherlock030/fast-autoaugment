import copy
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch

import numpy as np
from hyperopt import hp
import ray
import gorilla
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.suggest.hyperopt import HyperOptSearch
#from ray.tune.suggest import HyperOptSearch # function removed
from ray.tune import register_trainable, run_experiments
from ray.tune import run # by zh
from tqdm import tqdm

from FastAutoAugment.archive import remove_deplicates, policy_decoder
from FastAutoAugment.augmentations import augment_list
from FastAutoAugment.common import get_logger, add_filehandler
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.metrics import Accumulator
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.train import train_and_eval
from theconf import Config as C, ConfigArgumentParser 


top1_valid_by_cv = defaultdict(lambda: list)


def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, 'step')

    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_top1_acc = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])
    print('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')
    return original(self)


patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)


logger = get_logger('Fast AutoAugment')


def _get_path(dataset, model, tag):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/%s_%s_%s.model' % (dataset, model, tag))     # TODO


@ray.remote(num_gpus=4, max_calls=1)
def train_model_remote(config, dataroot, augment, cv_ratio_test, cv_fold, save_path=None, skip_exist=False):
    print('into training')
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment

    result = train_and_eval(None, dataroot, cv_ratio_test, cv_fold, save_path=save_path, only_eval=skip_exist)
    return C.get()['model']['type'], cv_fold, result

def train_model(config, dataroot, augment, cv_ratio_test, cv_fold, save_path=None, skip_exist=False):
    print('into training')
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment

    result = train_and_eval(None, dataroot, cv_ratio_test, cv_fold, save_path=save_path, only_eval=skip_exist)
    return C.get()['model']['type'], cv_fold, result


def eval_tta(config, augment, reporter):
    C.get()
    C.get().conf = config
    cv_ratio_test, cv_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']

    # setup - provided augmentation rules
    C.get()['aug'] = policy_decoder(augment, augment['num_policy'], augment['num_op'])

    # eval
    model = get_model(C.get()['model'], num_class(C.get()['dataset']))
    ckpt = torch.load(save_path)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    loaders = []
    for _ in range(augment['num_policy']):  # TODO
        _, tl, validloader, tl2 = get_dataloaders(C.get()['dataset'], C.get()['batch'], augment['dataroot'], cv_ratio_test, split_idx=cv_fold)
        loaders.append(iter(validloader))
        del tl, tl2

    start_t = time.time()
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    try:
        while True:
            losses = []
            corrects = []
            for loader in loaders:
                data, label = next(loader)
                data = data.cuda()
                label = label.cuda()

                pred = model(data)

                loss = loss_fn(pred, label)
                losses.append(loss.detach().cpu().numpy())

                _, pred = pred.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                corrects.append(correct)
                del loss, correct, pred, data, label

            losses = np.concatenate(losses)
            losses_min = np.min(losses, axis=0).squeeze()

            corrects = np.concatenate(corrects)
            corrects_max = np.max(corrects, axis=0).squeeze()
            metrics.add_dict({
                'minus_loss': -1 * np.sum(losses_min),
                'correct': np.sum(corrects_max),
                'cnt': len(corrects_max)
            })
            del corrects, corrects_max
    except StopIteration:
        pass

    del model
    metrics = metrics / 'cnt'
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    reporter(minus_loss=metrics['minus_loss'], top1_valid=metrics['correct'], elapsed_time=gpu_secs, done=True)
    return metrics['correct']


if __name__ == '__main__':
    import json
    from pystopwatch2 import PyStopwatch
    w = PyStopwatch() # 初始化一个秒表

    # ? 命令里面的 -c xxx.yaml不知道在哪里定义的
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--until', type=int, default=5) # ?
    parser.add_argument('--num-op', type=int, default=2) # 每个子策略里面的op数量
    parser.add_argument('--num-policy', type=int, default=5) # 每个policy里面包含5个子策略 
    parser.add_argument('--num-search', type=int, default=200) # ?还不确定,论文里写是每次贝叶斯优化的策略集合B的大小
    parser.add_argument('--cv-ratio', type=float, default=0.4) # ?交叉验证的比例
    parser.add_argument('--decay', type=float, default=-1) # ?可能是学习率衰减
    parser.add_argument('--redis', type=str, default='gpu-cloud-vnode30.dakao.io:23655') # 分布式相关的
    parser.add_argument('--per-class', action='store_true') # ?
    parser.add_argument('--resume', action='store_true') # ?应该是是否复用模型的参数吧
    parser.add_argument('--smoke-test', action='store_true') # ?
    parser.add_argument('--remote', action='store_true', help = 'whether to use distributed training')
    args = parser.parse_args()
    #print('args is ', args)
    #sys.exit(0)

    # 当命令行参数中存在有效的decay，就使用它
    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay

    add_filehandler(logger, os.path.join('models', '%s_%s_cv%.1f.log' % (C.get()['dataset'], C.get()['model']['type'], args.cv_ratio))) #logger添加file handler
    logger.info('configuration...')
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4)) #print config
    logger.info('initialize ray...') 
    #sys.exit(0)
    if args.remote:
        ray.init(redis_address=args.redis) # 启动分布式

    num_result_per_cv = 10 # ? 可能是交叉验证那个
    #cv_num = 5 # ? 可能是交叉验证那个
    cv_num = 1 # ! temp change
    copied_c = copy.deepcopy(C.get().conf) #copy一份config

    logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    logger.info('----- Train without Augmentations cv=%d ratio(test)=%.1f -----' % (cv_num, args.cv_ratio))
    w.start(tag='train_no_aug') # 秒表开始，no_aug训练
    paths = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_fold%d' % (args.cv_ratio, i)) for i in range(cv_num)] #5个模型的存储path
    print(paths)
    #sys.exit(0)
    # 训练no_aug模型
    # 底下的计算是对应在ray.get(reqs),
    # 代码逻辑是，如果save_path有效，就从save_path读取然后训练，训练结束存储在save_path里
    
    if args.remote:
        reqs = [
            train_model.remote(copy.deepcopy(copied_c), args.dataroot, C.get()['aug'], args.cv_ratio, i, save_path=paths[i], skip_exist=True)
            for i in range(cv_num)]
    else:
        reqs = [
            train_model(copy.deepcopy(copied_c), args.dataroot, C.get()['aug'], args.cv_ratio, i, save_path=paths[i], skip_exist=True)
            for i in range(cv_num)]

    #sys.exit(0)

    # ? 这边到底在干嘛，感觉没有用
    # ! 暂时全部注释
    
    tqdm_epoch = tqdm(range(C.get()['epoch'])) # tqdm是加载progress bar
    print('c epoch is ',C.get()['epoch'] )
    is_done = False
    for epoch in tqdm_epoch:
        print('epoch_', epoch, '_started')
        while True: # ! while true只是导致了死循环，没有别的了
            epochs_per_cv = OrderedDict() # 输出时按照输入顺序的字典
            for cv_idx in range(cv_num):
                try:
                    latest_ckpt = torch.load(paths[cv_idx]) # 加载一个刚训练好的模型
                    #print('latest ckpt,', latest_ckpt)
                    if 'epoch' not in latest_ckpt:
                        print('not in')
                        epochs_per_cv['cv%d' % (cv_idx + 1)] = C.get()['epoch']
                        continue
                    epochs_per_cv['cv%d' % (cv_idx+1)] = latest_ckpt['epoch']
                except Exception as e:
                    print('exception_', cv_idx)
                    continue
            tqdm_epoch.set_postfix(epochs_per_cv)
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= C.get()['epoch']:
                print('is done set')
                is_done = True
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= epoch:
                print('break inner loop')
                break
            time.sleep(10)
        if is_done:
            print('break outter loop')
            break
    logger.debug('useless code finished')
    #sys.exit(0)
    

    logger.info('getting results...')
    if args.remote:
        pretrain_results = ray.get(reqs)
    else:
        pretrain_results = reqs
    for r_model, r_cv, r_dict in pretrain_results:
        logger.info('model=%s cv=%d top1_train=%.4f top1_valid=%.4f' % (r_model, r_cv+1, r_dict['top1_train'], r_dict['top1_valid']))
    #print('watch is ', w.pause('train_no_aug'))
    w.pause('train_no_aug')
    # ! watch的结果是none，需要查阅api
    #logger.info('processed in %.4f secs' % w) 
    #sys.exit(0)
    
    if args.until == 1: #? 不知道的参数
        sys.exit(0)

    logger.info('----- Search Test-Time Augmentation Policies -----')
    w.start(tag='search')

    ops = augment_list(False) # op操作列表，false去掉了最后几个op
    space = {} # 用于超参数搜索，结合hp包使用
    # ? 5个policy会不会太少
    # hp.choice或者hp.uniform只是代表了一个搜索空间
    for i in range(args.num_policy): # args.num_policy 5
        for j in range(args.num_op): # args.num_policy 2
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)
    print('policy is , ', space['policy_0_0'])
    print('prob is , ', space['prob_0_0'])
    print('level is , ', space['level_0_0'])

    # sys.exit(0)

    final_policy_set = []
    total_computation = 0
    reward_attr = 'top1_valid'      # top1_valid or minus_loss
    for _ in range(1):  # run multiple times.
        for cv_fold in range(cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (C.get()['dataset'], C.get()['model']['type'], cv_fold, args.cv_ratio)
            print(name)
            try:
                register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_c), augs, rpt)) # 原代码写法
            except:
                pass
            #try:
            #    register_trainable(name, lambda config, reporter: eval_tta(copy.deepcopy(config), augs, reporter)) # 写法1,暂时可以运行
            #except:
            #    pass
            # algo = HyperOptSearch(space, max_concurrent=4*20, reward_attr=reward_attr) #! 函数过时了，reward_attr 可能是metric
            algo = HyperOptSearch(space, max_concurrent=4*20, metric=reward_attr)

            exp_config = {
                name: {
                    'run': name,
                    'num_samples': 4 if args.smoke_test else args.num_search,
                    'resources_per_trial': {'gpu': 1},
                    'stop': {'training_iteration': args.num_policy},
                    'config': {
                        'dataroot': args.dataroot, 'save_path': paths[cv_fold],
                        'cv_ratio_test': args.cv_ratio, 'cv_fold': cv_fold,
                        'num_op': args.num_op, 'num_policy': args.num_policy
                    },
                }
            }
            results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, resume=args.resume, raise_on_failed_trial=False) # 参数过时了
            #results = run(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, resume=args.resume, raise_on_failed_trial=False)

            results = [x for x in results if x.last_result is not None]
            results = sorted(results, key=lambda x: x.last_result[reward_attr], reverse=True)

            # calculate computation usage
            for result in results:
                total_computation += result.last_result['elapsed_time']

            for result in results[:num_result_per_cv]:
                final_policy = policy_decoder(result.config, args.num_policy, args.num_op)
                logger.info('loss=%.12f top1_valid=%.4f %s' % (result.last_result['minus_loss'], result.last_result['top1_valid'], final_policy))

                final_policy = remove_deplicates(final_policy)
                final_policy_set.extend(final_policy)
    

    logger.info(json.dumps(final_policy_set))
    logger.info('final_policy=%d' % len(final_policy_set))
    w.pause('search')
    print(w)
    #logger.info('processed in %.4f secs, gpu hours=%.4f' % (w.pause('search'), total_computation / 3600.))
    logger.info('----- Train with Augmentations model=%s dataset=%s aug=%s ratio(test)=%.1f -----' % (C.get()['model']['type'], C.get()['dataset'], C.get()['aug'], args.cv_ratio))
    #? 看起来在这里以上就把policy搜索完了？
    sys.exit(0)
    
    w.start(tag='train_aug')

    num_experiments = 5
    default_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_default%d' % (args.cv_ratio, _)) for _ in range(num_experiments)]
    augment_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_augment%d' % (args.cv_ratio, _)) for _ in range(num_experiments)]
    print('default path is ', default_path)
    print('augment path is ', augment_path)
    if args.remote:
        reqs = [train_model.remote(copy.deepcopy(copied_c), args.dataroot, C.get()['aug'], 0.0, 0, save_path=default_path[_], skip_exist=True) for _ in range(num_experiments)] + \
            [train_model.remote(copy.deepcopy(copied_c), args.dataroot, final_policy_set, 0.0, 0, save_path=augment_path[_]) for _ in range(num_experiments)]
    else:
        reqs = [train_model(copy.deepcopy(copied_c), args.dataroot, C.get()['aug'], 0.0, 0, save_path=default_path[_], skip_exist=True) for _ in range(num_experiments)] + \
            [train_model(copy.deepcopy(copied_c), args.dataroot, final_policy_set, 0.0, 0, save_path=augment_path[_]) for _ in range(num_experiments)]

    
    #? 又来了，这段丝毫看不出意义的code，似乎只是确认了存在一些已训练好的模型而且epoch数足够
    tqdm_epoch = tqdm(range(C.get()['epoch'])) # progress bar
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(num_experiments):
                try:
                    if os.path.exists(default_path[exp_idx]):
                        latest_ckpt = torch.load(default_path[exp_idx])
                        epochs['default_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass
                try:
                    if os.path.exists(augment_path[exp_idx]):
                        latest_ckpt = torch.load(augment_path[exp_idx])
                        epochs['augment_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass

            tqdm_epoch.set_postfix(epochs)
            if len(epochs) == num_experiments*2 and min(epochs.values()) >= C.get()['epoch']:
                is_done = True
            if len(epochs) == num_experiments*2 and min(epochs.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    if args.remote:
        final_results = ray.get(reqs)
    else:
        final_results = reqs

    for train_mode in ['default', 'augment']:
        avg = 0.
        for _ in range(num_experiments):
            r_model, r_cv, r_dict = final_results.pop(0)
            logger.info('[%s] top1_train=%.4f top1_test=%.4f' % (train_mode, r_dict['top1_train'], r_dict['top1_test']))
            avg += r_dict['top1_test']
        avg /= num_experiments
        logger.info('[%s] top1_test average=%.4f (#experiments=%d)' % (train_mode, avg, num_experiments))
    #logger.info('processed in %.4f secs' % w.pause('train_aug'))

    logger.info(w)
