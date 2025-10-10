#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=20:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Pretrain_basic

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"

cd diffusion_basic

python3 train.py --results_folder './concat_down4_prob01' --use_T2W --down 4 --blank_prob 0.1

date