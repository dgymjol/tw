dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32

gpunum=1



results_root='result_length_aug_'


list="2025 2024 2023 2022 2021 2020 2019 2018 2017 2016"

for seed in $list
do
  echo $seed

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python taskweave/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id lad_bothaug_mcls_4_${seed} \
--crop \
--merge \
--thres_crop 10 \
--thres_merge 30 \
--m_classes "[12, 36, 65, 150]" \
--tgt_embed \
--cc_matching \
--seed ${seed} \
${@:1}

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python taskweave/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id lad_bothaug_mcls_3_1_${seed} \
--crop \
--merge \
--thres_crop 10 \
--thres_merge 30 \
--m_classes "[8.33, 26.60, 56, 150]" \
--tgt_embed \
--cc_matching \
--seed ${seed} \
${@:1}

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python taskweave/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id lad_bothaug_mcls_3_2_${seed} \
--crop \
--merge \
--thres_crop 10 \
--thres_merge 30 \
--m_classes "[14.50, 31, 61.67, 150]" \
--tgt_embed \
--cc_matching \
--seed ${seed} \
${@:1}

done

