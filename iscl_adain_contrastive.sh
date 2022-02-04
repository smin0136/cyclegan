docker run --gpus '"device=1"' --rm -v /home:/home -ti smin/cycle:2.2.0 python \
	                  /home/Alexandrite/leekanggeun/MICCAI2022/MRI_Recon/cyclegan/iscl_adain_contrastive.py  \
			               --output_date='0202' --dir_num='4'  \
				       --adversarial_loss_mode='lsgan' \
			               --epochs=200 \

