# Copyright (c) OpenMMLab. All rights reserved.
import os
import argparse
import random
import json
import functools
import imagesize
from pathlib import Path

from hand_craft_translator import Translator
import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
#import pycocotools.mask as maskUtils

import ipdb


def collect_files(img_paths, gt_dir):
    
    files = []
    for img_file in img_paths:
        
        inst_file = gt_dir / img_file.with_suffix(".json").name
        assert inst_file.exists(), "annotaiton file does not exist"
        segm_file = ""
        files.append((img_file, inst_file, segm_file))

    return files


def create_img_soft_links(files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def _f(i):
        img_file, inst_file, segm_file = i
        cmd = f"ln -s {img_file} {out_dir / img_file.name}"
        if not (out_dir / img_file.name).exists():
            os.system(cmd)
        
    mmcv.track_progress(_f, files)
        

def collect_annotations(files, translator, nproc=1):
    print('Loading annotation images')

    f = functools.partial(load_img_info, translator=translator)
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            f, files, nproc=nproc)
    else:
        images = mmcv.track_progress(f, files)
        
    return images


def load_img_info(files, translator):

    img_file, inst_file, segm_file = files
    
    img_w, img_h = imagesize.get(img_file)
    labels = json.load(inst_file.open())
    
    anno_info = []
    for k, v in labels["instance"]["bbox2d"].items():
        x_top_left = v["xmin"] * img_w
        y_top_left = v["ymin"] * img_h
        
        bbox_width = (v["xmax"] - v["xmin"]) * img_w
        bbox_height = (v["ymax"] - v["ymin"]) * img_h
        
        category_id = labels["instance"]["class"][k]
        truncated = labels["instance"]["truncated"][k]
        occluded = labels["instance"]["occluded"][k]
        if translator.if_drop(bbox=[x_top_left, y_top_left, bbox_width, bbox_height], 
                              img_w=img_w, 
                              img_h=img_h, 
                              category_id=category_id, 
                              truncated=truncated, 
                              occluded=occluded
                             ):
            continue
        
        bbox = translator.shift_or_scale(bbox=[x_top_left, y_top_left, bbox_width, bbox_height], 
                          img_w=img_w, 
                          img_h=img_h, 
                          category_id=category_id, 
                          truncated=truncated, 
                          occluded=occluded
                         )
        
        
        if category_id >= 24:
            anno = dict(
                iscrowd=0,
                category_id=category_id,
                truncated=truncated, 
                occluded=occluded, 
                bbox=bbox,
                area=bbox_width * bbox_height,
                segmentation=dict(size=[0,0], counts="")
            )
            anno_info.append(anno)
        
    
    img_info = dict(
        # remove img_prefix for filename
        file_name=img_file.name,
        segm_file=segm_file, 
        height=img_h,
        width=img_w,
        anno_info=anno_info,
    )
    
    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in CSLabels.labels:
        if label.hasInstances and not label.ignoreInEval:
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Synscapes annotations to COCO format')
    parser.add_argument('synscapes_path', help='synscapes data path')
    
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    parser.add_argument("--drop", type=str, default="no", help="[param]-[criterion(small,truncated,occluded)]")
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--n', default=1000, help='Number of images sampled')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    synscapes_path = Path(args.synscapes_path)
    out_dir = Path(args.out_dir) if args.out_dir else Path(synscapes_path)
    mmcv.mkdir_or_exist(out_dir)

    img_dir = synscapes_path / "img/rgb"
    gt_dir = synscapes_path / "meta"

    img_paths = [p for p in img_dir.glob("*.png") if not p.name.startswith("._")]
    img_paths.sort()
    
    random.seed(args.seed)
    random.shuffle(img_paths)
    if args.reverse:
        img_paths = img_paths[::-1]
    
    translator = Translator(args.shift, args.scale, args.drop)
    
    set_name = [('train', f'annotations/bbox_train_{args.seed}_{args.n}.json'),
                ('val', f'annotations/bbox_val_{args.seed}_{args.n}.json')]

    for split, json_name in set_name:
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It took {}s to convert Synscapes annotation'):
            files = collect_files(img_paths[:args.n], gt_dir)
            img_paths = img_paths[args.n:]
            
            create_img_soft_links(files, out_dir / split / "rgb")
            image_infos = collect_annotations(files, translator=translator, nproc=args.nproc)
            
            
            cvt_annotations(image_infos, out_dir / split / json_name)


if __name__ == '__main__':
    main()


