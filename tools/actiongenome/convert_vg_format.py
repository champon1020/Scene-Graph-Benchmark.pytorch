import argparse
import json
import pickle
import h5py
import os.path as op
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--dst_dir", type=str, default="./datasets/ag")
args = parser.parse_args()

label_to_idx, idx_to_label = {}, {}
with open(op.join(args.root_dir, "annotations/object_classes.txt"), "r") as fp:
    lines = fp.readlines()
    class_id = 1
    for line in lines:
        label_to_idx[line.rstrip()] = class_id
        idx_to_label[str(class_id)] = line.rstrip()
        class_id += 1

predicate_to_idx, idx_to_predicate = {}, {}
with open(op.join(args.root_dir, "annotations/relationship_classes.txt"), "r") as fp:
    lines = fp.readlines()
    class_id = 1
    for line in lines:
        predicate_to_idx[line.rstrip()] = class_id
        idx_to_predicate[str(class_id)] = line.rstrip()
        class_id += 1

ag_sgg_dicts = {
    "idx_to_label": idx_to_label,
    "label_to_idx": label_to_idx,
    "idx_to_predicate": idx_to_predicate,
    "predicate_to_idx": predicate_to_idx,
    "attribute_to_idx": {},
    "idx_to_attribute": {},
}

with open(op.join(args.root_dir, "annotations/frame_list.txt"), "r") as fp:
    lines = fp.readlines()
    frame_list = [line.rstrip() for line in lines]

person_bbox = pickle.load(
    open(op.join(args.root_dir, "annotations/person_bbox.pkl"), "rb")
)

object_bbox_and_relationship = pickle.load(
    open(op.join(args.root_dir, "annotations/object_bbox_and_relationship.pkl"), "rb")
)

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


for frame_id in tqdm(frame_list):
    persons = person_bbox[frame_id]
    object_and_relationships = object_bbox_and_relationship[frame_id]

    if len(persons["bbox"]) == 0:
        continue

    img_to_first_box.append(bbox_count)
    img_to_first_rel.append(rel_count)

    # convert to (cxcywh)
    p_box = xyxy_to_cxcywh(persons["bbox"][0])
    boxes.append(np.array(p_box, dtype=np.int32))
    labels.append(np.array([label_to_idx["person"]], dtype=np.int32))
    w, h = persons["bbox_size"]
    image_data.append({"width": w, "height": h, "image_id": frame_id[:-4]})
    person_box_id = bbox_count
    bbox_count += 1

    image_set = 0
    for obj_and_rel in object_and_relationships:
        if obj_and_rel["bbox"] is None == 0:
            continue

        if (
            obj_and_rel["attention_relationship"] is None
            or obj_and_rel["spatial_relationship"] is None
            or obj_and_rel["contacting_relationship"] is None
        ):
            continue

        all_predicates = (
            obj_and_rel["attention_relationship"]
            + obj_and_rel["spatial_relationship"]
            + obj_and_rel["contacting_relationship"]
        )
        if len(all_predicates) == 0:
            continue

        obj_box = xywh_to_cxcywh(obj_and_rel["bbox"])
        obj_box = np.array(obj_box, dtype=np.int32)
        obj_label = np.array(
            [label_to_idx[obj_and_rel["class"].replace("/", "")]], dtype=np.int32
        )
        sub_obj = np.array([person_box_id, bbox_count], dtype=np.int32)
        boxes.append(obj_box)
        labels.append(obj_label)
        bbox_count += 1

        for rel in all_predicates:
            pred = np.array([predicate_to_idx[rel.replace("_", "")]], dtype=np.int32)
            relationships.append(sub_obj)
            predicates.append(pred)
            rel_count += 1

        image_set = 0 if obj_and_rel["metadata"]["set"] == "train" else 2

    split.append(image_set)

    img_to_last_box.append(bbox_count - 1)
    img_to_last_rel.append(rel_count - 1)


attributes = [np.array([0], dtype=np.int32) for _ in range(len(labels))]


with open(op.join(args.dst_dir, "VG-SGG-dicts.json"), "w") as fp:
    json.dump(ag_sgg_dicts, fp)

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
