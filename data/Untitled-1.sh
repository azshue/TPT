python ./tta_bongard.py /bongard --train_set bongard --bongard_split unseen_obj_unseen_act --ctx_init a_photo_of_a -a RN50 --pgen coop -b 64 --crop_scale 0.5 --severity 1.0 --aug_ops empty --tta_steps 1 --tta_lr 5e-3 --tta_optim AdamW --gpu 0 -p 10 --savedir temp
            
            