utils_files=$(wildcard /home/ceng/public/torchutils/torchutils/**/*.py)
env_path=/home/ceng/public/torchutils/envs/test-torchutils
install_path=${env_path}/lib/python3.6/site-packages/torchutils

all: ${install_path}

${install_path}: ${utils_files}
	@-pip uninstall -y torchutils
	@pip install -U ./

test_trainer: ${install_path}
	#cd tests/ && python3 -c 'import torchutils'
	python3 -m pytest ./tests/$@/

resnet: ${install_path}
	python3 ./test_resnet/main.py

.PHONY: tests resnet
