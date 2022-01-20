docker run --gpus '"device=1"' --rm -v /home:/home -ti smin/cycle:2.2.0 python \
	        /home/Alexandrite/smin/cycle_git/pre_train.py  \
		        --output_date='0118' --dir_num='1'  \
			        --adversarial_loss_mode='lsgan' \
				--epochs=200 \



