.PHONY: pretrain_ag
pretrain_ag:
	python -m torch.distributed.launch --master_port 10025 --nproc_per_node=4 tools/detector_pretrain_net.py \
				--config-file "configs/ag/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" \
				SOLVER.IMS_PER_BATCH 8 \
				TEST.IMS_PER_BATCH 4 \
				DTYPE "float16" \
				SOLVER.MAX_ITER 20000 \
				SOLVER.STEPS "(30000, 45000)" \
				SOLVER.VAL_PERIOD 10000 \
				SOLVER.CHECKPOINT_PERIOD 10000 \
				MODEL.RELATION_ON False \
				OUTPUT_DIR ./checkpoint/pretrained_faster_rcnn \
				SOLVER.PRE_VAL False

.PHONY: train_motif
train_motif:
	python -m torch.distributed.launch --master_port 10025 --nproc_per_node=4 tools/relation_train_net.py \
        --config-file "configs/ag/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
        SOLVER.IMS_PER_BATCH 12 \
        TEST.IMS_PER_BATCH 4 \
				SOLVER.PRE_VAL False \
        DTYPE "float16" \
        SOLVER.MAX_ITER 30000 \
        SOLVER.VAL_PERIOD 10000 \
        SOLVER.CHECKPOINT_PERIOD 10000 \
        GLOVE_DIR ./ASE/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/model_final.pth \
        OUTPUT_DIR ./results/motif

.PHONY: train_vctree
train_vctree:
	python -m torch.distributed.launch --master_port 10025 --nproc_per_node=4 tools/relation_train_net.py \
        --config-file "configs/ag/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
        SOLVER.IMS_PER_BATCH 12 \
        TEST.IMS_PER_BATCH 4 \
				SOLVER.PRE_VAL False \
        DTYPE "float16" \
        SOLVER.MAX_ITER 30000 \
        SOLVER.VAL_PERIOD 10000 \
        SOLVER.CHECKPOINT_PERIOD 10000 \
        GLOVE_DIR ./ASE/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/model_final.pth \
        OUTPUT_DIR ./results/vctree

.PHONY: train_imp
train_imp:
	python -m torch.distributed.launch --master_port 10025 --nproc_per_node=4 tools/relation_train_net.py \
        --config-file "configs/ag/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor \
        SOLVER.IMS_PER_BATCH 12 \
        TEST.IMS_PER_BATCH 4 \
				SOLVER.PRE_VAL False \
        DTYPE "float16" \
        SOLVER.MAX_ITER 30000 \
        SOLVER.VAL_PERIOD 10000 \
        SOLVER.CHECKPOINT_PERIOD 10000 \
        GLOVE_DIR ./ASE/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/model_final.pth \
        OUTPUT_DIR ./results/imp

.PHONY: train_causal
train_causal:
	python -m torch.distributed.launch --master_port 10025 --nproc_per_node=4 tools/relation_train_net.py \
        --config-file "configs/ag/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
        SOLVER.IMS_PER_BATCH 12 \
        TEST.IMS_PER_BATCH 4 \
				SOLVER.PRE_VAL False \
        DTYPE "float16" \
        SOLVER.MAX_ITER 30000 \
        SOLVER.VAL_PERIOD 10000 \
        SOLVER.CHECKPOINT_PERIOD 10000 \
        GLOVE_DIR ./ASE/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/model_final.pth \
        OUTPUT_DIR ./results/causal
