docker run --gpus '"device=2"' --rm -v /home:/home -ti smin/iscl_neu:2.8.0 python \
	                  /home/Alexandrite/smin/cycle_git/mri_proposed.py  \
			               --output_date='0524' --dir_num='2'  \
				       --adversarial_loss_mode='lsgan'  \
			               --epochs=200 --epoch_decay=100 \
				       --crop_size=256 --load_size=256 \

