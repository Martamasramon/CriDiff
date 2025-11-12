import os 
from log_functions import (
     parse_config_block, 
     save_config_json, 
     parse_log_metrics, 
     save_metrics_csv,
     log_csv_json_to_wandb
     )


def create_wandb_log(log_path):
     args_dict = parse_config_block(log_path)
     
     name = args_dict['results_folder'][2:]
     out_json = name + ".json"
     save_config_json(args_dict, out_json) 
     print(f"Wrote {len(args_dict)} keys to {out_json}")
     
     out_csv = name +  ".csv"
     rows = parse_log_metrics(log_path)
     save_metrics_csv(rows, out_csv)
     print(f"Wrote {len(rows)} rows to {out_csv}")
     
     log_csv_json_to_wandb(name+".csv", name+".json")
     
     
if __name__ == '__main__':   
     paths = [i for i in os.listdir('./') if '620059' in i]

     for path in paths:
        print(path)
        try:
             create_wandb_log('./'+ path)
        except Exception as e:
             print(e)
