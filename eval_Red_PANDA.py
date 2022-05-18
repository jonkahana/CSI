from common.eval import *
import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os

exps_dir = '/cs/labs/yedid/jonkahana/projects/Red_PANDA/cache/models'
model_dir = f'unsup_simclr_CSI_{P.dataset}_{P.model}_shift_rotation_one_class_0'
model_dir = os.path.join(exps_dir, P.dataset, model_dir)
model_checkpoint_name = 'last.model'

P.load_path = os.path.join(model_dir, model_checkpoint_name)
checkpoint = torch.load(P.load_path)
model.load_state_dict(checkpoint, strict=not P.no_strict)
model.eval()

from evals.ood_pre_Red_PANDA import eval_ood_detection


def Red_PANDA_knn_score(train_set, test_set, n_neighbours=2):
    index_flat = faiss.IndexFlatL2(train_set.shape[1])
    # res = faiss.StandardGpuResources()  # use a single GPU
    # index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index = index_flat
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def Red_PANDA_knn_inds(train_set, test_set, n_neighbours=5):
    index_flat = faiss.IndexFlatL2(train_set.shape[1])
    # res = faiss.StandardGpuResources()  # use a single GPU
    # index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index = index_flat
    index.add(train_set)
    D, I = index.search(test_set, n_neighbours)
    return I


def Red_PANDA_get_score(test_scores, test_labels):

    if P.eval_psuedo:
        #### pseudo_anom scoring
        pseudo_anom_test_inds = np.where(test_labels == 2)[0]
        pseudo_anom_test_lbls = np.zeros(len(test_labels))
        pseudo_anom_test_lbls[pseudo_anom_test_inds] = 1

        pseudo_auc = roc_auc_score(pseudo_anom_test_lbls, test_scores)

        #### pseudo_anom only scoring
        no_anom_test_inds = np.where(test_labels != 1)[0]
        no_anom_test_labels = deepcopy(test_labels[no_anom_test_inds])
        pseudo_only_anom_test_inds = np.where(no_anom_test_labels == 2)[0]
        pseudo_only_anom_test_lbls = np.zeros(len(no_anom_test_labels))
        pseudo_only_anom_test_lbls[pseudo_only_anom_test_inds] = 1

        pseudo_only_auc = roc_auc_score(pseudo_only_anom_test_lbls, test_scores[no_anom_test_inds])

        #### anom w.r.t pseudo scoring
        anom_wrt_psuedo_test_lbls_indxs = np.where(test_labels >= 1)[0]
        scores_anom_and_pseudo = test_scores[anom_wrt_psuedo_test_lbls_indxs]
        anom_and_pseudo_test_labels = test_labels[anom_wrt_psuedo_test_lbls_indxs]
        anom_wrt_psuedo_ROC_lbls = np.zeros(len(anom_and_pseudo_test_labels))
        anom_wrt_psuedo_ROC_lbls[np.where(anom_and_pseudo_test_labels == 1)[0]] = 1

        anom_auc_wrt_psuedo = roc_auc_score(anom_wrt_psuedo_ROC_lbls, scores_anom_and_pseudo)
    else:
        pseudo_auc, anom_auc_wrt_psuedo, pseudo_only_auc = -1., -1., -1.

    #### anom scoring
    anom_test_inds = np.where(test_labels == 1)[0]
    anom_test_lbls = np.zeros(len(test_labels))
    anom_test_lbls[anom_test_inds] = 1

    anom_auc = roc_auc_score(anom_test_lbls, test_scores)

    ### avg. consistency
    # todo: implement

    if not P.eval_psuedo:
        scores_out = f"FIXED || ROC-AUC: {anom_auc:.3f}"
    else:
        scores_out = f"FIXED || ROC-AUC: {anom_auc:.3f} || Pseudo Anom AUC {pseudo_auc:.3f} || " \
                     f"|| Pseudo vs. Normal AUC {pseudo_only_auc:.3f} || Anom w.r.t Pseudo AUC {anom_auc_wrt_psuedo:.3F}"
    print(scores_out)

    return anom_auc, pseudo_auc, anom_auc_wrt_psuedo, pseudo_only_auc


with torch.no_grad():
    test_scores, test_targets = eval_ood_detection(P, model, test_loader,
                                                   train_loader=train_loader,
                                                   simclr_aug=simclr_aug)
    test_targets = np.array(test_targets)
    test_scores *= -1.
    # test_scores = np.random.uniform(0, 1, size=(test_scores.shape[0])) #todo remove

print(test_scores)
print(test_scores.shape)
print(test_targets.shape)
print(np.unique(test_targets))


anom_auc, pseudo_auc, anom_auc_wrt_psuedo, psuedo_vs_normal_auc = Red_PANDA_get_score(test_scores, test_targets)
results_df = pd.DataFrame(np.zeros((1, 5)),
                          index=['ours'],
                          columns=['scoring_method', 'anom_auc', 'pseudo_auc',
                                   'anom_auc_wrt_psuedo', 'psuedo_vs_normal_auc'])
df_row = {'anom_auc': anom_auc, 'pseudo_auc': pseudo_auc, 'scoring_method':'knn',
          'anom_auc_wrt_psuedo': anom_auc_wrt_psuedo, 'psuedo_vs_normal_auc': psuedo_vs_normal_auc}
for k in df_row.keys():
    results_df.loc['ours', k] = df_row[k]

os.makedirs(os.path.join(model_dir, 'vanilla'), exist_ok=True)
results_df.to_csv(os.path.join(model_dir, 'vanilla', 'auc_results.csv'))

