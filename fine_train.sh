docker run --gpus '"device=0"' --rm -v /home:/home -ti smin/cycle:2.2.0 python \
	  /home/Alexandrite/smin/cycle_git/fine_train.py  \
	  --output_date='0118' --dir_num='5'  \
	   --adversarial_loss_mode='lsgan' \
	  --epochs=100 \





