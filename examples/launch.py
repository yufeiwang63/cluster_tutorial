import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from examples.train import run_task

def get_lr(epoch):
    if epoch == 10:
        return 1e-2
    else:
        return 1e-3

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'test_experiment'

    vg = VariantGenerator()

    # dressing env args
    vg.add('seed', [100, 200]) # 2 runs with different seeds
    vg.add('epoch', [10, 20]) # train for 10 or 20 epochs
    vg.add('lr', lambda epoch: [get_lr(epoch)]) # the value of 1 variable can depend on the value(s) of another variable(s)
    vg.add('cuda_id', [0]) # the value of 1 variable can depend on the value(s) of another variable(s)

    if debug:
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()
    gpu_num = torch.cuda.device_count()

    variations = set(vg.variations())
    task_per_gpu = 1 # how many tasks you wanna run on each slurm allocated GPU
    all_vvs = vg.variants()
    slurm_nums = len(all_vvs) // task_per_gpu
    if len(all_vvs) % task_per_gpu != 0:
        slurm_nums += 1

    # now launch the job on seuss
    sub_process_popens = []
    for idx in range(slurm_nums):
        beg = idx * task_per_gpu
        end = min((idx+1) * task_per_gpu, len(all_vvs))
        vvs = all_vvs[beg:end]
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss']:
            if idx == 0:
                compile_script = None  
                wait_compile = 0
            else:
                compile_script = None
                wait_compile = 0  
        else:
            compile_script = wait_compile = None
        env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variants=vvs,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var,
            variations=variations,
            task_per_gpu=task_per_gpu
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
