import argparse
import h5py
import json
import os
import os.path as op
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--dst_dir", type=str, default="./datasets/vidvrd")
args = parser.parse_args()

label_to_idx, idx_to_label = {}, {}
with open(op.join(args.root_dir, "object.txt")) as fp:
    lines = fp.readlines()
    class_id = 1
    for line in lines:
        label_to_idx[line.rstrip()] = class_id
        idx_to_label[str(class_id)] = line.rstrip()
        class_id += 1

predicate_to_idx, idx_to_predicate = {}, {}
with open(op.join(args.root_dir, "predicate.txt")) as fp:
    lines = fp.readlines()
    relationship_classes = [line.rstrip() for line in lines]
    class_id = 1
    for line in lines:
        predicate_to_idx[line.rstrip()] = class_id
        idx_to_predicate[str(class_id)] = line.rstrip()
        class_id += 1

sgg_dicts = {
    "idx_to_label": idx_to_label,
    "label_to_idx": label_to_idx,
    "idx_to_predicate": idx_to_predicate,
    "predicate_to_idx": predicate_to_idx,
    "attribute_to_idx": {},
    "idx_to_attribute": {},
}

videos = os.listdir(op.join(args.root_dir, "videos"))

image_data = []
bbox_count, rel_count = 0, 0
img_to_first_box, img_to_last_box = [], []
img_to_first_rel, img_to_last_rel = [], []
boxes, labels = [], []
relationships, predicates = [], []
split = []


def xyxy_to_cxcywh(bbox):
    # convert to xywh
    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    return xywh_to_cxcywh(bbox)


def xywh_to_cxcywh(bbox):
    return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]


for video_name in tqdm(videos):
    video_id = op.splitext(video_name)[0]
    split_name = (
        "train"
        if op.exists(op.join(args.root_dir, "train", f"{video_id}.json"))
        else "test"
    )
    split_number = 0 if split_name == "train" else 2

    annot = json.load(open(op.join(args.root_dir, split_name, f"{video_id}.json")))
    w, h = annot["width"], annot["height"]
    tid_to_label = {o["tid"]: o["category"] for o in annot["subject/objects"]}

    relation_instances = [[] for _ in range(len(annot["trajectories"]))]
    for rel in annot["relation_instances"]:
        for fid in range(rel["begin_fid"], rel["end_fid"]):
            relation_instances[fid].append(
                {
                    "sub": rel["subject_tid"],
                    "rel": predicate_to_idx[rel["predicate"]],
                    "obj": rel["object_tid"],
                }
            )

    for fid, objs in enumerate(annot["trajectories"]):
        frame_id = f"{video_id}.mp4/{fid:06d}.jpg"
        rels = relation_instances[fid]

        if len(objs) == 0 or len(rels) == 0:
            continue

        img_to_first_box.append(bbox_count)
        img_to_first_rel.append(rel_count)

        tid_to_obj_idx = {}
        for obj in objs:
            bbox = xyxy_to_cxcywh(
                [
                    obj["bbox"]["xmin"],
                    obj["bbox"]["ymin"],
                    obj["bbox"]["xmax"],
                    obj["bbox"]["ymax"],
                ]
            )
            label = label_to_idx[tid_to_label[obj["tid"]]]
            boxes.append(np.array(bbox, dtype=np.int32))
            labels.append(np.array([label], dtype=np.int32))
            tid_to_obj_idx[obj["tid"]] = bbox_count
            bbox_count += 1

        for rel in rels:
            relationship = [tid_to_obj_idx[rel["sub"]], tid_to_obj_idx[rel["obj"]]]
            predicate = rel["rel"]
            relationships.append(np.array(relationship, dtype=np.int32))
            predicates.append(np.array([predicate], dtype=np.int32))
            rel_count += 1

        image_data.append({"width": w, "height": h, "image_id": frame_id[:-4], "ext": "jpg"})
        split.append(split_number)
        img_to_last_box.append(bbox_count - 1)
        img_to_last_rel.append(rel_count - 1)

attributes = [np.array([0], dtype=np.int32) for _ in range(len(labels))]


with open(op.join(args.dst_dir, "VG-SGG-dicts.json"), "w") as fp:
    json.dump(sgg_dicts, fp)

with open(op.join(args.dst_dir, "image_data.json"), "w") as fp:
    json.dump(image_data, fp)

with h5py.File(op.join(args.dst_dir, "VG-SGG.h5"), "w") as fp:
    fp.create_dataset("img_to_first_box", data=img_to_first_box)
    fp.create_dataset("img_to_last_box", data=img_to_last_box)
    fp.create_dataset("img_to_first_rel", data=img_to_first_rel)
    fp.create_dataset("img_to_last_rel", data=img_to_last_rel)
    fp.create_dataset("boxes_1024", data=boxes)
    fp.create_dataset("labels", data=labels)
    fp.create_dataset("predicates", data=predicates)
    fp.create_dataset("relationships", data=relationships)
    fp.create_dataset("attributes", data=attributes)
    fp.create_dataset("split", data=split)
