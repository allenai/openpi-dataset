import json
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

with open("../data/dev-data-reformatted-v4_votes_score_salience_1-20.json") as f_gold, open("../data/dev-data-reformatted-v4_votes_salience_1-20_human2.json") as f_gold2, open("../data/dev-data-reformatted-v4_pred-salience.json") as f_pred:
    gold = json.load(f_gold)
    gold2 = json.load(f_gold2)
    pred = json.load(f_pred)
    pearsonrs_g_p = []
    pearsonrs_g_v = []
    pearsonrs_g_s = []
    pearsonrs_g_g2 = []
    pearsonrs_g_p_local = []
    pearsonrs_g_v_local = []
    pearsonrs_g_s_local = []
    pearsonrs_g_g2_local = []
    for (id, g_proc), (id, g2_proc), (id, p_proc) in zip(gold.items(), gold2.items(), pred.items()):
    #for (id, g_proc), (id, g2_proc) in zip(gold.items(), gold2.items()):
        if id == "21":
            break
        gold_salience = []
        gold2_salience = []
        vote_salience = []
        pred_salience = []
        score_salience = []
        gold_salience_local = []
        gold2_salience_local = []
        vote_salience_local = []
        pred_salience_local = []
        score_salience_local = []
        #print(g_proc["steps"])
        for g_state, g2_state, p_state in zip(g_proc["states"], g2_proc["states"], p_proc["states"]):
            # global
            g_state["global_salience"] = int(g_state["global_salience"])
            g2_state["global_salience"] = int(g2_state["global_salience"])
            p_state["global_salience_pred"] = int(p_state["global_salience_pred"])
            vote_salience.append(g_state["votes"])
            score_salience.append(g_state["score"])
            if g_state["global_salience"] in [0,1,2,3,4,5]:
                gold_salience.append(g_state["global_salience"])
            else:
                gold_salience.append(0)
            if g2_state["global_salience"] in [0,1,2,3,4,5]:
                gold2_salience.append(g2_state["global_salience"])
            else:
                gold2_salience.append(0)
            if p_state["global_salience_pred"] in [0,1,2,3,4,5]:
                pred_salience.append(p_state["global_salience_pred"])
            else:
                pred_salience.append(0)
            # local
            for (_, g_step), (_, g2_step), (_, p_step)  in zip(g_state["answers"].items(), g2_state["answers"].items(), p_state["answers"].items()):
                #print(g_step)
                gold_salience_local.append(g_step["local_salience"])
                gold2_salience_local.append(int(g2_step["local_salience"]))
                vote_salience_local.append(g_step["votes"])
                score_salience_local.append(g_step["score"])
                pred_salience_local.append(int(p_step["local_salience_pred"]))

        assert(len(gold_salience) == len(pred_salience))

        # Smoothing
        vote_salience = np.append(vote_salience, 0)
        vote_salience = 5 * (vote_salience-np.min(vote_salience))/(np.max(vote_salience)-np.min(vote_salience))
        vote_salience_local = np.append(vote_salience_local, 0)
        vote_salience_local = 5 * (vote_salience_local-np.min(vote_salience_local))/(np.max(vote_salience_local)-np.min(vote_salience_local))

        score_salience = np.append(score_salience, 0)
        score_salience = 5 * (score_salience-np.min(score_salience))/(np.max(score_salience)-np.min(score_salience))
        score_salience_local = np.append(score_salience_local, 0)
        score_salience_local = 5 * (score_salience_local-np.min(score_salience_local))/(np.max(score_salience_local)-np.min(score_salience_local))

        gold_salience.append(0)
        gold2_salience.append(0)
        pred_salience.append(0)
        gold_salience_local.append(0)
        gold2_salience_local.append(0)
        #print(gold_salience_local)
        #print(gold2_salience_local)
        pred_salience_local.append(0)

        corr_g_g2 = pearsonr(gold_salience, gold2_salience)[0]
        corr_g_g2_local = pearsonr(gold_salience_local, gold2_salience_local)[0]
        corr_g_p = pearsonr(gold_salience, pred_salience)[0]
        corr_g_p_local = pearsonr(gold_salience_local, pred_salience_local)[0]
        corr_g_v = pearsonr(gold_salience, vote_salience)[0]
        corr_g_v_local = pearsonr(gold_salience_local, vote_salience_local)[0]
        corr_g_s = pearsonr(gold_salience, score_salience)[0]
        corr_g_s_local = pearsonr(gold_salience_local, score_salience_local)[0]

        #print(gold_salience_local, gold2_salience_local, corr_g_g2_local)
        pearsonrs_g_p.append(corr_g_p)
        pearsonrs_g_p_local.append(corr_g_p_local)
        pearsonrs_g_g2.append(corr_g_g2)
        pearsonrs_g_g2_local.append(corr_g_g2_local)
        pearsonrs_g_v.append(corr_g_v)
        pearsonrs_g_v_local.append(corr_g_v_local)
        pearsonrs_g_s.append(corr_g_s)
        pearsonrs_g_s_local.append(corr_g_s_local)

print(pearsonrs_g_s_local)
print(np.mean(pearsonrs_g_s_local))