docker run --gpus '"device=0"' --rm -v /home:/home -ti smin/cycle:2.2.0 python \
	                 /home/Alexandrite/smin/cycle_git/pre_res.py  \
			                         --output_date='0123' --dir_num='2'  \
						 --adversarial_loss_mode='lsgan' \
						 --epochs=200 \

