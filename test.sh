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

source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"

cd diffusion_attn
python3 test.py --checkpoint './pretrain/model-best.pt'  --use_T2W --save_name 'pretrain_blank_t2w'
python3 test.py --checkpoint './pretrain_down4/model-best.pt'  --use_T2W --down 4
python3 test.py --checkpoint './pretrain_down8/model-best.pt'  --use_T2W --down 8

cd ../diffusion_basic
python3 test.py --checkpoint './pretrain/model-8.pt' 
python3 test.py --checkpoint './pretrain_mask/model-best.pt' --use_mask 
python3 test.py --checkpoint './pretrain_down4/model-best.pt' --down 4
python3 test.py --checkpoint './pretrain_down8/model-best.pt' --down 8
python3 test.py --checkpoint './test_concat/model-best.pt' --use_T2W

python3 test.py --checkpoint './concat_down4_prob01/model-best.pt' --use_T2W --down 4
python3 test.py --checkpoint './concat_down4_prob03/model-best.pt' --use_T2W --down 4
python3 test.py --checkpoint './concat_down8/model-best.pt' --use_T2W --down 8
python3 test.py --checkpoint './concat_down8_prob02/model-best.pt' --use_T2W --down 8

python3 test.py --checkpoint './pretrain_down8_mask/model-best.pt' --down 8 --use_mask

date

