#!/bin/bash
# Table 3 (policy transfer): every ImageNet, ADE20K and COCO-Stuff model evaluated with heuristic
# policies and with the learned policies of the three datasets (same seed index as the model).
# The native cells are those of Table 2 and are shared with table2.sh.
#   bash reproduce/table3.sh [datasets]      default: imagenet ade20k coco_stuff
# Then: python reproduce/aggregate.py
cd "$(dirname "$0")/.." && source reproduce/common.sh
for DS in ${1:-imagenet ade20k coco_stuff}; do
  # the map heuristic of each task: feature saliency (classification), prediction entropy (segmentation)
  if [ "$DS" = imagenet ]; then MAP=saliency; else MAP=entropy; fi
  for ARM in random learned; do
    for S in $SEEDS; do
      cell "$DS" "$ARM" "$S" random       4 --policy random
      cell "$DS" "$ARM" "$S" triangle     4 --policy triangle
      cell "$DS" "$ARM" "$S" quadrant     5 --policy quadrant
      cell "$DS" "$ARM" "$S" "${MAP}_ior" 4 --policy "$MAP" --ior_radius 0.5
      for PDS in imagenet ade20k coco_stuff; do
        if [ "$ARM" = learned ] && [ "$PDS" = "$DS" ]; then
          cell "$DS" learned "$S" learned 4 --policy learned   # native, as in Table 2
        else
          transplant "$DS" "$ARM" "$S" "$PDS"
        fi
      done
    done
  done
done
