project_prefix=/home/ceng/public/torchutils
utils_files=$(wildcard ${project_prefix}/torchutils/**/*.py)
env_path=${project_prefix}/envs/torchutils
install_path=${env_path}/lib/python3.6/site-packages/torchutils

all: ${install_path} docs

${install_path}: ${utils_files}
	@-pip uninstall -y torchutils
	@python3 setup.py build
	@python3 setup.py install --skip-build

docs:  docs/man/man3/_home_ceng_public_torchutils_torchutils_.3


docs/man/man3/_home_ceng_public_torchutils_torchutils_.3: ${utils_files} CHANGELOG.md README.md Doxyfile
	@doxygen ./Doxyfile

test_trainer: ${install_path}
	#cd tests/ && python3 -c 'import torchutils'
	python3 -m pytest ./tests/$@/

resnet: ${install_path}
	python3 ./test_resnet/main.py

.PHONY: tests resnet
