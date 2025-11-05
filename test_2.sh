#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=00:59:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Test

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

cd diffusion_basic
python3 test.py --checkpoint './concat_down8/model-best.pt' --use_T2W --down 8 --t2w_offset 5
python3 test.py --checkpoint './concat_down8/model-best.pt' --use_T2W --down 8 --t2w_offset 15
python3 test.py --checkpoint './concat_down8/model-best.pt' --use_T2W --down 8 --t2w_offset 20
python3 test.py --checkpoint './concat_down8/model-best.pt' --use_T2W --down 8 --t2w_offset 50

date

