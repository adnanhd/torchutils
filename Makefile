utils_files=$(wildcard /home/ceng/public/torchutils/torchutils/**/*.py)
env_path=/home/ceng/public/cnnfoil/envs/cnnfoil
install_path=${env_path}/lib/python3.6/site-packages/torchutils

all: ${install_path}

${install_path}: ${utils_files}
	@-pip uninstall -y torchutils
	@pip install -U ./

tests: ${install_path}
	#cd tests/ && python3 -c 'import torchutils'
	python3 -m pytest ./tests/

.PHONY: tests
