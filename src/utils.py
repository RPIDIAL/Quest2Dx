import pandas as pd
import re, os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
from sklearn import linear_model
from released.genomics_utils_legacy import clean_wgs_snpid
import subprocess
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.functional import log_softmax
from torch import Tensor, topk
from torch import erf, as_tensor, eye, zeros, int64, float32, arange, tensor
from tqdm import trange
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.cuda import set_device
import colorsys

### General Utils ###
def darken_color(color, amount=0.5):
    c = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def select_gpu(num_gpus=1,verbose=False):
    # Run the nvidia-smi command to get GPU information
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], capture_output=True, text=True)

    # Parse the output to get GPU index and memory usage
    gpu_info = result.stdout.strip().split('\n')
    gpu_info = [info.split(',') for info in gpu_info]
    gpu_info = [(info[0], int(info[1].split()[0])) for info in gpu_info]

    # Sort the GPU info based on memory usage
    sorted_gpu_info = sorted(gpu_info, key=lambda x: x[1])

    if verbose:
        # Print the GPU info with least memory usage
        for gpu in sorted_gpu_info:
            print(f"GPU {gpu[0]}: Memory Usage {gpu[1]} MB")
    
    # Select the first num_gpus GPUs with least memory usage
    selected_gpus = [gpu[0] for gpu in sorted_gpu_info[:num_gpus]]
    return selected_gpus

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

## clinical_dataset utils
def imputer(df):
    # TODO: implement alternative imputers (mean/median)
    # Impute and fill missing values using MICE (sklearn implementation)
    mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, imputation_order='ascending', keep_empty_features = True)
    # Temporarily filling columns with all missing values using 0s, so that number of columns is preserved.
    imp = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)
    return imp

def filter_and_imput(df_full,visit,demo_vars,clin_vars,outcome_vars,impute,vis_colnm):
    # Filter at visit 
    df_full = df_full.loc[df_full[vis_colnm] == visit,:]
    
    # drop column for visit ID
#     df_full = df_full.drop('EVENT_ID', axis =1)
    df_full = df_full[df_full[vis_colnm].notna()]
    
    # Replace empty rows that have ' ' for nan
    df_full = df_full.replace(r'^\s*$', np.nan, regex=True)

    # Drop rows if all values are nan
    df_full = df_full.dropna(axis = 0, how = 'all')

    # Filter cols / vars of interest
    df_demo = df_full.loc[:,demo_vars]
    df_x = df_full.loc[:,clin_vars]
    df_y = df_full.loc[:,outcome_vars]

    # remove all non numeric charcters
    df_x = df_x.replace(r'\D', np.nan, regex=True)
    
#     Impute X
    if impute:
        df_x = imputer(df_x)

    # Set ID as index for all 3 dfs
    df_demo = df_demo.set_index('PTID') # Set PTID as index in df
    df_x.index = df_demo.index.values.astype('str')
    df_y.index = df_demo.index.values.astype('str')
    return df_demo,df_x,df_y

## genomics_dataset utils
# TODO: merge clean pat ids into just one function based on if there are "_" for ADNI or not (for PPMI -> "I")
def clean_pat_wgs_id_ADNI(x):
    split_id = re.split('_', x)
    return split_id[len(split_id)-1]

def clean_pat_wgs_id(x):
    split_id = re.split('I', x)
    return split_id[len(split_id)-1]

def wgs_cleaner(wgs):
    wgs = wgs.loc[:,wgs.columns.difference(['IID','PAT','MAT','SEX', 'PHENOTYPE'])]
    wgs['FID'] = wgs.FID.apply(lambda x: clean_pat_wgs_id_ADNI(x)) # Remember to switch between ADNI and PPMI
    wgs = wgs.rename(columns={'FID':'PATNO'})
    wgs = wgs.set_index('PATNO')
    wgs = wgs.astype('float')
    # TODO: switch imputation by column mean to better imputation method (currently only 10 NAs though)
    wgs = wgs.fillna(wgs.mean())
    wgs = wgs.astype('int')
    # wgs = wgs.loc[~wgs.PATNO.duplicated()]
    # Rename columns to only SNP RS id
    wgs_snp_ids = wgs.columns
    # wgs_snp_ids = wgs_snp_ids[1:].map(lambda x: clean_wgs_snpid(x)) # because we used to have the phenotype column in there
    wgs_snp_ids = wgs_snp_ids.map(lambda x: clean_wgs_snpid(x))
    # wgs_snp_ids = wgs_snp_ids.insert(0,'PATNO')
    wgs.columns = wgs_snp_ids
    wgs = wgs.loc[:,~wgs.columns.duplicated()]
    return wgs

def select_snps(genes,gene):
    genes = genes.reset_index()
    return genes.loc[genes.Gene == gene].SNV.values # SNPs are now index and not column
    # return genes.loc[genes.MAPPED_GENE == gene].index.values

def subset_snps_betas(df_snps,betas,snplist):
    df_snps_filtered = df_snps.loc[:,df_snps.columns.intersection(snplist)]
    betas_filtered = betas.loc[betas.index.isin(df_snps_filtered.columns.values)]
    return df_snps_filtered, betas_filtered

def gene_rep(df_snps,betas_dropdup,snplist):
    snps_filtered, betas_filtered = subset_snps_betas(df_snps,betas_dropdup,snplist)
    return np.dot(snps_filtered.values,betas_filtered.values)


## img_dataset utils
def filter_and_impute_img(df_full,visit,vis_colnm,cols_interest,impute):
    # Filter at visit 
    df_full = df_full.loc[df_full[vis_colnm] == visit,:]
    
    # drop column for visit ID
#     df_full = df_full.drop('EVENT_ID', axis =1)
    df_full = df_full[df_full[vis_colnm].notna()]
    
    # Filter cols / vars of interest
    df_full = df_full.loc[:,cols_interest]

    # Replace empty rows that have ' ' for nan
    df_full = df_full.replace(r'^\s*$', np.nan, regex=True)

    # Drop rows if all values are nan
    df_full = df_full.dropna(axis = 0, how = 'all')
    
    # Drop columns if all values are nan
    # df_full = df_full.dropna(axis=1, how='all') # Option removed so that the tokenization process works
    # Impute
    if impute:
        df_full_imp = imputer(df_full)
         # Reset index
        df_full_imp.index = df_full.index
        return df_full_imp
    return df_full


### Results utils
def compute_auc(output,target):
    # FIXME: currently if batch has only one class return AUC of 0.5
    if len(np.unique(np.round(output[:,0]))) == 1 and output.shape[1] == 2 and len(np.unique(np.round(target))) != 1:
        return 0.5
    
    if len(np.unique(np.round(output[:,0]))) == 1 and len(np.unique(np.round(target))) == 1:
        return 1.0
    
    if len(np.unique(np.round(output[:,0]))) == 2 and len(np.unique(np.round(target))) == 1: # FIXME: this is not correct, but patch for now.
        return 1.0

    if len(np.unique(target)) == 2:
        micro_roc_auc_ovo = roc_auc_score(
            target,
            output[:,1]
        )
    else:
        micro_roc_auc_ovo = roc_auc_score(
            target,
            output,
            multi_class="ovo",
            average="macro",
            labels=np.asarray([0,1,2])
        )

    return micro_roc_auc_ovo

def compute_auc_pr(output, target):
    """
    Compute the Area Under the Precision-Recall Curve (AUC-PR).

    Parameters:
    output (array): Output scores, where each row corresponds to a sample and each column to a class.
    target (array): True binary labels.

    Returns:
    float: AUC-PR value.
    """
    # Handle edge cases
    if len(np.unique(np.round(output[:, 0]))) == 1 and output.shape[1] == 2 and len(np.unique(np.round(target))) != 1:
        return 0.5

    if len(np.unique(np.round(output[:, 0]))) == 1 and len(np.unique(np.round(target))) == 1:
        return 1.0

    if len(np.unique(np.round(output[:, 0]))) == 2 and len(np.unique(np.round(target))) == 1:
        return 1.0

    # Calculate AUC-PR for binary classification
    if len(np.unique(target)) == 2:
        auc_pr = average_precision_score(target, output[:, 1])
    # Calculate AUC-PR for multi-class classification
    else:
        auc_pr = average_precision_score(
            target,
            output,
            average="macro"
        )

    return auc_pr

def results_summary(results_dict, fname_root_out, itr):
    metrics =[]
    for i in range(10):
        for j in range(5):
            metrics.append(pd.DataFrame.from_dict([results_dict[i][j]['metrics']]))

    # TODO: concatenate the metrics dataframe per iteration into 1
    merged_df = pd.concat(metrics)

    # Per test set results 
    df_per_test_set = per_test_set_summary(results_dict)

    # Save results dict and dataframe
    merged_df.to_csv(str('../results/'+fname_root_out+'_iter_'+str(itr)+'_results_summary.csv'), index=False)

    df_per_test_set.to_csv(str('../results/'+fname_root_out+'_iter_'+str(itr)+'_results_summary_per_testset.csv'), index=False)

    with open(str('../results/'+fname_root_out+'_iter_'+str(itr)+'_results_dictionary.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)
    return merged_df


def per_test_set_summary(results_dict_all_splits):
    df_res = []
    colnames = ['loss train avg','loss train std','loss val avg','loss val std','loss test avg','loss test std',
                'AUC train avg','AUC train std','AUC val avg','AUC val std','AUC test avg','AUC test std',]
    for i in results_dict_all_splits.keys():
        loss_train_l = []
        loss_val_l = []
        loss_test_l = []
        auc_train_l = []
        auc_val_l = []
        auc_test_l = []
        for j in results_dict_all_splits[i]:
            loss_train_l.append(results_dict_all_splits[i][j]['metrics']['loss_train'])
            loss_val_l.append(results_dict_all_splits[i][j]['metrics']['loss_val'])
            loss_test_l.append(results_dict_all_splits[i][j]['metrics']['loss_test'])
            auc_train_l.append(results_dict_all_splits[i][j]['metrics']['auc_train'])
            auc_val_l.append(results_dict_all_splits[i][j]['metrics']['auc_val'])
            auc_test_l.append(results_dict_all_splits[i][j]['metrics']['auc_test'])
        df_res.append(pd.DataFrame(np.asarray((np.mean(loss_train_l), np.std(loss_train_l), np.mean(loss_val_l),
                                               np.std(loss_val_l), np.mean(loss_test_l), np.std(loss_test_l),
                                               np.mean(auc_train_l), np.std(auc_train_l), np.mean(auc_val_l),
                                              np.std(auc_val_l), np.mean(auc_test_l), np.std(auc_test_l))).reshape(1,12), columns = colnames))
            
    df_res = pd.concat(df_res)
    df_res.loc[len(df_res.index)] = df_res.mean().values
    df_res.loc[len(df_res.index)] = df_res.std().values
    return df_res


def results_display(results_df, metric_name):

    mean_metric = results_df[metric_name].mean()

    print('#'*40)
    print('Final results:')
    print(f'The average {metric_name} across all iterations (repeated experiments) is {mean_metric}')


def plot_coattn(dict_path,quant_res_path,clin_feat_path,img_feat_path,gen_feat_path,k_feat):
    # load results dictionary
    with open(dict_path, 'rb') as handle:
        data = handle.read()
    d = pickle.loads(data)
  
    # load quantitavie results summary table
    quant_res = pd.read_csv(quant_res_path)
    quant_res.auc_test.idxmax()

    # load feature names
    clin_feat = np.loadtxt(clin_feat_path, dtype='str').tolist()
    img_feat = np.loadtxt(img_feat_path, dtype='str').tolist()
    # gen_feat = np.loadtxt(gen_feat_path, dtype='str', delimiter="$").tolist() # Many gene names are not parsed properly so needed to add fake delimiter to allow for loading
    gen_feat = np.loadtxt(gen_feat_path, dtype='str').tolist()

    # find top performing model
    best_model_idx = quant_res.auc_test.idxmax()

    # Plot best performing model's attention accross subjects
    attn_plot_mod('img_clin',best_model_idx,d,clin_feat,img_feat,gen_feat,k_feat)
    attn_plot_mod('img_gen',best_model_idx,d,clin_feat,img_feat,gen_feat,k_feat)
    
    
def attn_plot_mod(mod,best_model_idx,d,clin_feat,img_feat,gen_feat,k_feat):
    if mod == 'img_clin':
        key = 'out_img_clin_attn_l_test'
        row_names = clin_feat
        col_names = img_feat
    elif mod == 'img_gen':
        key = 'out_img_gen_attn_l_test'
        row_names = gen_feat 
        col_names = img_feat
    # choose only attn scores from top performing model
    attn_best_model = d[best_model_idx//5][best_model_idx-((best_model_idx//5)*5)]['data'][key]

    # Average accross patients in top performing model and
    attn_mat_avg = np.mean(np.squeeze(np.asarray(attn_best_model), axis=1), axis = 0)

    # Max signal thresholding and formating
    if len(row_names) < k_feat:
        k_feat = len(row_names)
    if len(col_names) < k_feat:
        k_feat = len(col_names)
    vals, i = topk(Tensor(attn_mat_avg.flatten()), k_feat)
    top_coords = np.array(np.unravel_index(i.numpy(), attn_mat_avg.shape)).T
    attn_mat_avg[attn_mat_avg < np.min(vals.numpy())] = 0
    attn_mat_avg_top = attn_mat_avg[top_coords[:,0],:]
    attn_mat_avg_top = attn_mat_avg_top[:,top_coords[:,1]]
    filtered_row_names = np.asarray(row_names)[top_coords[:,0]]
    filtered_col_names = np.asarray(col_names)[top_coords[:,1]]
    df = pd.DataFrame(attn_mat_avg_top, index = filtered_row_names, columns = filtered_col_names)
    df = df.loc[:,~df.columns.duplicated()].copy()
    df = df[~df.index.duplicated(keep='first')]

    # Save attn to csv file
    df.to_csv(f'../results/attn_scores_{mod}_set_{best_model_idx}_top_k_feat_{k_feat}.csv')

    # fn_clin = str('../results/attn_scores_img_clin_set_'+str(best_model_idx)+'_top_k_feat_'+str(k_feat)+'.csv')
    # fn_gen = str('../results/attn_scores_img_gen_set_'+str(best_model_idx)+'_top_k_feat_'+str(k_feat)+'.csv')
    # fn_out_clin = str('../results/img_clin')
    # fn_out_gen = str('../results/img_gen')
    fn_mod = str('../results/attn_scores_'+str(mod)+'_set_'+str(best_model_idx)+'_top_k_feat_'+str(k_feat)+'.csv')
    fn_out = str('../results/figures/attn_plots/'+str(mod))

    plot_r_subprocess(fn_mod,fn_out)
        
       
def plot_r_subprocess(fn_mod,fn_out):
    cmd = 'conda run -n R_upd Rscript attn_plot.R n '+fn_mod+' '+fn_out
    subprocess.call(cmd.split(), shell = False)



#### New utils - includes tokenization of tabular data ####
# Obtained from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
def preprocess_img_no_tokenization(df,roi_df,visit,vis_colnm,cols_interest):
    # Filter at visit 
    df = df.loc[df[vis_colnm] == visit,:]
    
    # drop column for visit ID
    df = df[df[vis_colnm].notna()]
    
    # Filter cols / vars of interest
    df = df.loc[:,cols_interest]

    # Replace empty rows that have ' ' for nan
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Drop all columns that have all nans for patients
    df = df.dropna(axis = 1, how = 'all')

    # Drop all rows that have all nans for patients
    df = df.dropna(axis = 0, how = 'all')

    # Filter for only Cross-Sectional analysis
    roi_df = roi_df[roi_df['FLDNAME'].isin(cols_interest)]
    roi_df = roi_df.loc[roi_df['CRFNAME'] == 'Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)',:]

    # Parse regions of interest
    roi_df['ROI'] = roi_df['TEXT'].str.split('of',expand=True).iloc[:,1]
    roi_df['Trait'] = roi_df['TEXT'].str.split('of',expand=True).iloc[:,0]
    roi_df['Trait'] =  roi_df['Trait'].replace(r"^ +| +$", r"", regex=True)
    roi_df = roi_df.set_index('FLDNAME')
    roi_shared = roi_df.loc[roi_df['Trait'] == 'Cortical Thickness Average','ROI']
    return df


def plot_training_curves(train_losses, train_aucs, val_losses, val_aucs, path):
    # Plot training curves
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(train_aucs, label='Train')
    plt.plot(val_aucs, label='Validation')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(path)
    plt.close()

# Code adapted from On Embeddings for Numerical Features in Tabular Deep Learning
# https://github.com/Yura52/tabular-dl-num-embeddings/blob/main/bin/train4.py - commit e1401da
# subsample and and tree options not implemented
def idp_tokenization(device,count,n_num_features,seed,idx,X_num):
    subsample = None
    tree = None
    encoding = 'piecewise-linear'

    print('\nRunning bin-based encoding...')
    assert X_num is not None
    bin_edges = []
    _bins = {x: [] for x in X_num}
    _bin_values = {x: [] for x in X_num}
    rng = np.random.default_rng(seed)
    for feature_idx in trange(n_num_features):
        train_column = X_num['train'][:, feature_idx].copy()
        if subsample is not None:
            subsample_n = (
                subsample
                if isinstance(subsample, int)
                else int(subsample * D.size('train'))
            )
            subsample_idx = rng.choice(len(train_column), subsample_n, replace=False)
            train_column = train_column[subsample_idx]
        else:
            subsample_idx = None

        if tree is not None:
            _y = D.y['train']
            if subsample_idx is not None:
                _y = _y[subsample_idx]
            tree = (
                (DecisionTreeRegressor if D.is_regression else DecisionTreeClassifier)(
                    max_leaf_nodes=C.bins.count, **C.bins.tree
                )
                .fit(train_column.reshape(-1, 1), D.y['train'])
                .tree_
            )
            del _y
            tree_thresholds = []
            for node_id in range(tree.node_count):
                # the following condition is True only for split nodes
                # See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    tree_thresholds.append(tree.threshold[node_id])
            tree_thresholds.append(train_column.min())
            tree_thresholds.append(train_column.max())
            bin_edges.append(np.array(sorted(set(tree_thresholds))))
        else:
            feature_n_bins = min(count, len(np.unique(train_column)))
            quantiles = np.linspace(
                0.0, 1.0, feature_n_bins + 1
            )  # includes 0.0 and 1.0
            # bin_edges.append(np.unique(np.quantile(train_column, quantiles))) #NOTE: this line is the original implementation, but cannot deal with nan values, subsitute with the following line using np.nanquantile
            bin_edges.append(np.unique(np.nanquantile(train_column, quantiles)))

        for part in X_num:
            _bins[part].append(
                np.digitize(
                    X_num[part][:, feature_idx],
                    np.r_[-np.inf, bin_edges[feature_idx][1:-1], np.inf],
                ).astype(np.int32)
                - 1
            )
            if encoding == 'binary':
                _bin_values[part].append(np.ones_like(X_num[part][:, feature_idx]))
            elif encoding == 'piecewise-linear':
                feature_bin_sizes = (
                    bin_edges[feature_idx][1:] - bin_edges[feature_idx][:-1]
                )
                part_feature_bins = _bins[part][feature_idx]
                # NOTE: added line below to deal with nan values in dataset by setting them to the last bin, this allows for /feature_bin_sizes[part_feature_bins] to work
                part_feature_bins[part_feature_bins > bin_edges[feature_idx][1:].shape[0]-1] = bin_edges[feature_idx][1:].shape[0]-1
                _bin_values[part].append(
                    (
                        X_num[part][:, feature_idx]
                        - bin_edges[feature_idx][part_feature_bins]
                    )
                    / feature_bin_sizes[part_feature_bins]
                )
            else:
                assert encoding == 'one-blob'

    n_bins = max(map(len, bin_edges)) - 1

    bins = {
        k: as_tensor(np.stack(v, axis=1), dtype=int64, device=device)
        for k, v in _bins.items()
    }
    del _bins

    bin_values = (
        {
            k: as_tensor(np.stack(v, axis=1), dtype=float32, device=device)
            for k, v in _bin_values.items()
        }
        if _bin_values['train']
        else None
    )
    del _bin_values
    bin_edges = [tensor(x, dtype=float32, device=device) for x in bin_edges]
    print()

    assert bins is not None
    assert bins is not None
    assert n_bins is not None

    if encoding == 'one-blob':
        assert bin_edges is not None
        assert X_num is not None
        assert C.bins.one_blob_gamma is not None
        x = zeros(
            len(idx), D.n_num_features, n_bins, dtype=float32, device=device
        )
        for i in range(D.n_num_features):
            n_bins_i = len(bin_edges[i]) - 1
            bin_left_edges = bin_edges[i][:-1]
            bin_right_edges = bin_edges[i][1:]
            kernel_scale = 1 / (n_bins_i ** C.bins.one_blob_gamma)
            cdf_values = [
                0.5
                * (
                    1
                    + erf(
                        (edges[None] - X_num[part][idx, i][:, None])
                        / (kernel_scale * 2 ** 0.5)
                    )
                )
                for edges in [bin_left_edges, bin_right_edges]
            ]
            x[:, i, :n_bins_i] = cdf_values[1] - cdf_values[0]

    else:
        assert bin_values is not None
        bins_ = bins[part][idx]
        bin_mask_ = eye(n_bins, device=device)[bins_]
        x = bin_mask_ * bin_values[part][idx, ..., None]
        previous_bins_mask = arange(n_bins, device=device)[None, None].repeat(
            len(idx), n_num_features, 1
        ) < bins_.reshape(len(idx), n_num_features, 1)
        x[previous_bins_mask] = 1.0

    return x