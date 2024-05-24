import mindspore as ms

def init_LearngenePool(learngenepool, blk_length, init_mode):
    init_learngenepool = []
    if blk_length == 6:
        if init_mode == 'ours':
            ckpt_paths = [
                './distill/deit_base_patch16_224-deit_tiny6_patch16_224/LearngenePool_[front,end]/checkpoint.ckpt',
                './distill/deit_base_patch16_224-deit_small6_patch16_224/LearngenePool_[end]/checkpoint.ckpt',
                './distill/deit_base_patch16_224-deit_base6_patch16_224/LearngenePool_[end]/checkpoint.ckpt']
        if init_mode == 'snnet':
            ckpt_paths = [
                './train_results/scratch/tiny6/checkpoint.pth',  # instance.0.
                './train_results/scratch/small6/checkpoint.pth',  # instance.1.
                './train_results/scratch/base6/checkpoint.pth']  # instance.2.
    if blk_length == 9:
        if init_mode == 'ours':
            ckpt_paths = [
                './distill/deit_base_patch16_224-deit_tiny9_patch16_224/LearngenePool_[front,end]/checkpoint.ckpt',
                './distill/deit_base_patch16_224-deit_small9_patch16_224/LearngenePool_[end]/checkpoint.ckpt',
                './distill/deit_base_patch16_224-deit_base9_patch16_224/LearngenePool_[end]/checkpoint.ckpt']
        if init_mode == 'snnet':
            ckpt_paths = [
                './train_results/scratch/tiny9/checkpoint.pth',
                './train_results/scratch/small9/checkpoint.pth',
                './train_results/scratch/base9/checkpoint.pth']

    for id, (model, ckpt_path) in enumerate(zip(learngenepool, ckpt_paths)):
        if ckpt_path == "./distill/deit_base_patch16_224-deit_small6_patch16_224/LearngenePool_[end]/checkpoint.ckpt" or ckpt_path == "./distill/deit_base_patch16_224-deit_small9_patch16_224/LearngenePool_[end]/checkpoint.ckpt":
            init_learngenepool.append(model)
            continue
        param_dict = ms.load_checkpoint(ckpt_path)
        print(ckpt_path)

        if init_mode == 'snnet':
            new_param_dict = {}
            # model_dict = model.state_dict()
            key = 'instances.{}'.format(id)
            for k, v in param_dict.items():
                if key in k:
                    new_key = k[12:]
                    new_param_dict[new_key] = v
            # assert model_dict.keys() == new_param_dict.keys()
            # model_dict.update(new_param_dict)
            ms.load_param_into_net(model, param_dict)
        else:
            new_param_dict = {}
            for name, param in param_dict.items():
                new_name = name.replace('model.', '')
                new_param_dict[new_name] = param
            ms.load_param_into_net(model, new_param_dict)
        init_learngenepool.append(model)
    print('The Learngene Pool is been initialized by mode {}'.format(init_mode))
    return init_learngenepool





