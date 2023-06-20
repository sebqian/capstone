docker run --rm --gpus all --name capstone_container -it --shm-size=64g \
	-v /home/qian/capstone/codebase:/workspace/codebase \
    -v /mnt/d:/workspace/data \
	capstone_img:latest
