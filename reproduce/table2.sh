#!/bin/bash
# Table 2 (native policy evaluation): every random-location and learned-policy model, evaluated with
# the policy it was trained with at 1, 4 and 6 glimpses. Learned policies take the argmax of their
# probability map; random-location models draw fixations uniformly.
#   bash reproduce/table2.sh [datasets]      default: cmnist imagenet ade20k coco_stuff
# Then: python reproduce/aggregate.py
cd "$(dirname "$0")/.." && source reproduce/common.sh
for DS in ${1:-cmnist imagenet ade20k coco_stuff}; do
  for S in $SEEDS; do
    for T in 1 4 6; do
      cell "$DS" random  "$S" random  "$T" --policy random
      cell "$DS" learned "$S" learned "$T" --policy learned
    done
  done
done
