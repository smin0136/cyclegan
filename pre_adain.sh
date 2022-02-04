docker run --gpus '"device=0"' --rm -v /home:/home -ti smin/cycle:2.2.0 python \
	                         /home/Alexandrite/smin/cycle_git/pre_adain.py  \
			--output_date='0202' --dir_num='4'  \
			--adversarial_loss_mode='lsgan' \
			--epochs=200 \

