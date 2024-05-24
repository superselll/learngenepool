# The procedures of building descendant models

## Building the learngene pool with 18 instances
1. Training the auxiliary models with 9 blocks.
```python
python train.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-5 \
                  --output_dir YOURPATH \
                  --teacher_model  deit_base_patch16_224 \
                  --student_model deit_base9_patch16_224 \
                  --loss_pos [end]\
                  --data-path YOUR DATA PATH
```
```python
python train.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-4 \
                  --output_dir YOURPATH \
                  --teacher_model  deit_base_patch16_224 \
                  --student_model deit_small9_patch16_224 \
                  --loss_pos [end]\
                  --data-path YOUR DATA PATH
```
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-4 \
                  --output_dir YOURPATH \
                  --teacher_model  deit_base_patch16_224 \
                  --student_model deit_tiny9_patch16_224 \
                  --loss_pos [front, end]\
                  --data-path YOUR DATA PATH
```
2. Building the learngene pool and finetune it
```python
# init_stitch_mode = ours or snnet, 'ours' means to initialize the stitching layers by the proposed method and snnet means by Sn-Net
python main.py --batch-size 128 \
               --epoch 50 \
               --lr 5e-4 \
               --output_dir YOURPATH\
               --blk_length 9 \
               --init_stitch_mode [ours|snnet] \
               --init_learngenepool_mode ours \
               --ls_init True \
               --data-path YOUR DATA PATH
```

3. Building variable-sized models from the learngene pool
```python
python main.py --eval True \\
               --batch-size 128 \
               --output_dir YOURPATH \
               --blk_length 9 --init_stitch_mode [ours|snnet] \
               --init_learngenepool_mode ours \
               --ls_init True \
               --data-path YOUR DATA PATH
```

## Building the learngene pool with 12 instances
1. Training the auxiliary models with 6 blocks.
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-4 \
                  --output_dir YOURPATH \
                  --teacher_model  deit_base_patch16_224 \
                  --student_model deit_base6_patch16_224 \
                  --data-path YOUR DATA PATH
```
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-4 \
                  --output_dir YOURPATH \
                  --teacher_model  deit_base_patch16_224 \
                  --student_model deit_tiny6_patch16_224 \
                  --data-path YOUR DATA PATH
```
2 and 3 are the same as the procedures in learngene pool with 18 instances but change the hyper-parameters, blk_length, to 6. 
