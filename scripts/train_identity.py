import os
import datetime
import argparse

from pathlib import Path


def execute(cmd, dry_run=False):
    print(cmd)
    if not dry_run:
        os.system(cmd)


def dump_new_config(domain_A_name, domain_B_name):
    base_config_path = "configs/my_cfg/base_label_translation_synscapes_resnet.py"
    new_config_path = f"configs/my_cfg/_label_translation_synscapes_{domain_A_name}x{domain_B_name}_resnet.py"
    
    home_dir = Path(os.environ["HOME"])
    
    with open(base_config_path) as f:
        cont = f.read()
        lines = cont.split("\n")
        for i, l in enumerate(lines):
            if l == "domain_a = None":
                new_l = f"""domain_a = '{domain_A_name}'"""
                lines[i] = new_l
                
            if l == "domain_b = None":
                new_l = f"""domain_b = '{domain_B_name}'"""
                lines[i] = new_l
                
            if l == "dataroot = None":
                new_l = f"""dataroot = '{home_dir / "datasets/synscapes"}/'"""
                lines[i] = new_l
            
    cont = "\n".join(lines)
    with open(new_config_path, "w") as f:
        f.write(cont)
        
    return new_config_path

    
def main():

    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-local', action='store_true', help="whether to run in local machine")
    parser.add_argument("--samples_per_gpu", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    
    parser.add_argument('--disjoint', action='store_true')
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    parser.add_argument("--drop", type=str, default="no", help="[param]-[criterion(small,truncated,occluded)]")

    args = parser.parse_args()

    
    home_dir = Path(os.environ["HOME"])
    os.chdir("../")

    domain_A_name = "unchanged"
    print(f"Preparing Synscapes domain A {domain_A_name}")
    cmd = f"python tools/dataset_converters/synscapes.py /datasets/synscapes --out-dir {home_dir / 'datasets/synscapes/unchanged'} --nproc 8 --shift {args.shift} --scale {args.scale} --drop {args.drop}"
                    
    execute(cmd, dry_run=args.dry_run)
    
    
    domain_B_name = []
    if args.shift != "no":
        ratio, direction = args.shift.split("-")
        ratio = int(float(ratio)*100)
        domain_B_name.append(f"shift-{ratio}-{direction}")

    if args.scale != "no":
        ratio, direction = args.scale.split("-")
        ratio = int(float(ratio)*100)
        domain_B_name.append(f"scale-{ratio}-{direction}")

    if args.drop != "no":
        param, criterion = args.drop.split("-")
        if criterion == "small":
            param = int(param)
        elif criterion in ["truncated", "occluded"]:
            param = int(param*100)
        else:
            raise ValueError
        domain_B_name.append(f"drop-{param}-{criterion}")


    domain_B_name = "_".join(domain_B_name) if len(domain_B_name) > 0 else "unchanged"
    print(f"Preparing Synscapes domain B {domain_B_name}")
    
    
    cmd = f"python tools/dataset_converters/synscapes.py /datasets/synscapes --out-dir {home_dir / 'datasets/synscapes' / domain_B_name} --nproc 8 --shift {args.shift} --scale {args.scale} --drop {args.drop}"
    if args.disjoint:
        cmd += " --reverse"
    execute(cmd, dry_run=args.dry_run)

    
    
    print("Training")
    config = dump_new_config(domain_A_name, domain_B_name)
        
    
    #effective_batch_size = args.num_gpus * args.samples_per_gpu
    #base_lr = 1e-4      #  for 8 GPUs and 2 img/gpu
    #lr = base_lr * effective_batch_size / 16

    ct = datetime.datetime.now()
    wandb_name = f"{ct.year}.{ct.month}.{ct.day}.{ct.hour}.{ct.minute}.{ct.second}"
    cmd = f"python tools/train.py {config} --cfg-options data.workers_per_gpu=0"
    
    if not args.run_local:
        cmd += f" --work-dir {home_dir / 'results'}"

    execute(cmd, dry_run=args.dry_run)


if __name__ == '__main__':
    main()