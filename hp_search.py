import os

random_trials = 5

configs = {}
configs['base'] = {'width-mult': 1., 'kernel-size': 5, 'dropout': 0., 'double-layers': False}
configs['double'] = {'width-mult': 1., 'kernel-size': 5, 'dropout': 0., 'double-layers': True}
configs['double_w0.5'] = {'width-mult': 0.5, 'kernel-size': 5, 'dropout': 0., 'double-layers': True}
configs['w0.5'] = {'width-mult': 0.5, 'kernel-size': 5, 'dropout': 0., 'double-layers': False}
configs['w0.75'] = {'width-mult': 0.75, 'kernel-size': 5, 'dropout': 0., 'double-layers': False}
configs['w1.25'] = {'width-mult': 1.25, 'kernel-size': 5, 'dropout': 0., 'double-layers': False}
configs['w1.5'] = {'width-mult': 1.5, 'kernel-size': 5, 'dropout': 0., 'double-layers': False}
configs['w2'] = {'width-mult': 2, 'kernel-size': 5, 'dropout': 0., 'double-layers': False}
configs['k1'] = {'width-mult': 1., 'kernel-size': 1, 'dropout': 0., 'double-layers': False}
configs['k3'] = {'width-mult': 1., 'kernel-size': 3, 'dropout': 0., 'double-layers': False}
configs['k7'] = {'width-mult': 1., 'kernel-size': 7, 'dropout': 0., 'double-layers': False}
configs['dr0.05'] = {'width-mult': 1., 'kernel-size': 5, 'dropout': 0.05, 'double-layers': False}
configs['dr0.1'] = {'width-mult': 1., 'kernel-size': 5, 'dropout': 0.1, 'double-layers': False}
configs['dr0.2'] = {'width-mult': 1., 'kernel-size': 5, 'dropout': 0.2, 'double-layers': False}

if __name__ == '__main__':

    for exp_name in configs:
        cfg = configs[exp_name]

        command = 'python tf/train.py'
        command += ' --exp-dir experiments/tf/single-res/hp_search'
        command += ' --width-mult {}'.format(cfg['width-mult'])
        command += ' --kernel-size {}'.format(cfg['kernel-size'])
        command += ' --dropout {}'.format(cfg['dropout'])
        if cfg['double-layers']:
            command += ' --double-layers'

        for i in range(random_trials):
            command += ' --seed {}'.format(i)
            command += ' --exp-name {}_{}'.format(exp_name, i)
            os.system(command)