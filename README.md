# The procedures of building descendant models

## Building the learngene pool with 18 instances
1. Training the auxiliary models with 9 blocks.
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-5 \
                  --blk_length 9\
                  --output_dir YOURPATH \
                  --teacher_model  midmap_deit_base_patch16_224 \
                  --student_model midmap_deit_base9_patch16_224 \
                  --loss_pos [end]\
                  --data-path YOUR DATA PATH
                  --teacher_path YOUR DATA PATH
```
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-5 \
                  --blk_length 9\
                  --output_dir YOURPATH \
                  --teacher_model  midmap_deit_base_patch16_224 \
                  --student_model midmap_deit_small9_patch16_224 \
                  --loss_pos [end]\
                  --data-path YOUR DATA PATH
                  --teacher_path YOUR DATA PATH
```
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-5 \
                  --blk_length 9\
                  --output_dir YOURPATH \
                  --teacher_model  midmap_deit_base_patch16_224 \
                  --student_model midmap_deit_tiny9_patch16_224 \
                  --loss_pos [front, end]\
                  --data-path YOUR DATA PATH
                  --teacher_path YOUR DATA PATH
```
2. Building the learngene pool and finetune it
```python
python main.py --batch-size 128 \
               --epoch 50 \
               --lr 5e-5 \
               --output_dir YOURPATH\
               --blk_length 9 \
               --data-path YOUR DATA PATH
```

3. Building variable-sized models from the learngene pool
```python
python eval.py --blk_length 9 \
               --batch-size 128 \
               --data-path YOUR DATA PATH
```

## Building the learngene pool with 18 instances
1. Training the auxiliary models with 9 blocks.
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-5 \
                  --blk_length 6\
                  --output_dir YOURPATH \
                  --teacher_model  midmap_deit_base_patch16_224 \
                  --student_model midmap_deit_base6_patch16_224 \
                  --loss_pos [end]\
                  --data-path YOUR DATA PATH
                  --teacher_path YOUR DATA PATH
```
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-5 \
                  --blk_length 6\
                  --output_dir YOURPATH \
                  --teacher_model  midmap_deit_base_patch16_224 \
                  --student_model midmap_deit_small6_patch16_224 \
                  --loss_pos [end]\
                  --data-path YOUR DATA PATH
                  --teacher_path YOUR DATA PATH
```
```python
python distill.py --batch-size 256 \
                  --epochs 100 \
                  --num_workers 64 \
                  --lr 5e-5 \
                  --blk_length 6\
                  --output_dir YOURPATH \
                  --teacher_model  midmap_deit_base_patch16_224 \
                  --student_model midmap_deit_tiny6_patch16_224 \
                  --loss_pos [front, end]\
                  --data-path YOUR DATA PATH
                  --teacher_path YOUR DATA PATH
```
2. Building the learngene pool and finetune it
```python
python main.py --batch-size 128 \
               --epoch 50 \
               --lr 5e-5 \
               --output_dir YOURPATH\
               --blk_length 6 \
               --data-path YOUR DATA PATH
```

3. Building variable-sized models from the learngene pool
```python
python eval.py --blk_length 6 \
               --batch-size 128 \
               --data-path YOUR DATA PATH
