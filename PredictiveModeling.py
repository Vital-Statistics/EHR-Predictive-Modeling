# -*- coding: utf-8 -*-
"""
Created on Sun Jun  17 23:43:08 2024

@author: emrys y
"""
import numpy as np
import pandas as pd
import scipy.stats as stat
import math
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from datetime import timedelta
import causalinference as ci
from scipy.stats import fisher_exact

PROJECT_PATH = 'D:/Vital Statistics/metabolomics data/GITHUB PAGE prep/GitHub pages/predictive_modeling/'





import logging
import warnings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore')




def xval_train_V3(mod, X, Y, grp=None, folds=20, leaveOneOut=False, verbose=False, returnAUC=True):
    import random
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if type(X) == np.ndarray:
        X = pd.DataFrame(X, index=range(len(X)), columns=['X' + v for v in range(X.shape[1])])
        Y = pd.DataFrame(Y, index=range(len(Y)), columns=['Y' + v for v in range(Y.shape[1])])

    S = pd.concat([Y.rename('y'), X], axis=1)
    S['xval'] = 0.0
    cols = list(X)

    if grp is None:
        if not leaveOneOut:
            S['grp'] = pd.Series(random.choices(list(range(folds)), k=len(S)), index=S.index)
        else:
            S['grp'] = range(len(S))
    else:
        S['grp'] = grp
    
    # cross-validation and gather predictions
    y_true = []
    y_pred = []
    for i, g in enumerate(S.grp.unique()):
        if verbose:
            loopProgress(i, len(S.grp.unique()), 'x-val progress')
        ll = S.grp != g
        mod.fit(S.loc[ll, cols], S.loc[ll, 'y'])
        predictions = mod.predict_proba(S.loc[~ll, cols])[:, 1]

        if len(predictions) != S.loc[~ll, 'y'].shape[0]:
            print(f"Warning: Mismatch in prediction sizes for group {g}. Adjusting.")
        
        S.loc[~ll, 'xval'] = predictions

        # Store true vs. predicted values for AUC computation
        y_true.extend(S.loc[~ll, 'y'])
        y_pred.extend(predictions)
        
    if returnAUC:
        auc_score = roc_auc_score(y_true, y_pred)
        return S['xval'], auc_score, y_true, y_pred
    else:
        return S['xval']











def selectAndTrain_original_V3(X, y, lbl='Setting regression penalty', returnYhat=False, verbose=False, leaveOneOut=False, top_n=10):
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.utils import resample
    from scipy import stats as stat
    from scipy.stats import bootstrap
    from statsmodels.sandbox.stats.multicomp import multipletests
    from sklearn.linear_model import LogisticRegression

    nNan = pd.concat([X, y], axis=1).isna().sum(axis=1)
    nDrop = (nNan > 0).sum()
    if nDrop > 0:
        print(f'Dropping {nDrop} observations due to missingness.')
        X = X.loc[nNan == 0]
        y = y.loc[nNan == 0]
    
    # cross-validate different penalty strengths
    xvp = pd.DataFrame({'tt': np.arange(.025, 2, .025), 'cstat': .5, 'P-Value': 1, 'FDR': 1})
    best_y_true, best_y_pred = None, None
    bestYHat = None
    best_auc = 0.5
    
    for i, tt in enumerate(xvp.tt):
        loopProgress(i, len(xvp), lbl)
        mod = LogisticRegression(max_iter=100, penalty='l1', solver='liblinear', C=tt)
        yhat, auc_score, y_true, y_pred = xval_train_V3(mod, X, y, leaveOneOut=leaveOneOut, returnAUC=True)
        # xvp.loc[xvp.tt == tt, 'P-Value'] = float(stat.ranksums(yhat.loc[y == 0], yhat.loc[y == 1], alternative='less')[1])
        p_val = float(stat.ranksums(yhat.loc[y == 0], yhat.loc[y == 1], alternative='less')[1])
        xvp.loc[xvp.tt == tt, 'P-Value'] = p_val
        xvp.loc[xvp.tt == tt, 'cstat'] = auc_score

        if auc_score >= best_auc:
            best_auc = auc_score
            best_y_true = y_true
            best_y_pred = y_pred
            bestYHat = yhat

    xvp.sort_values('cstat', inplace=True, ascending=False)
    xvp['FDR'] = multipletests(xvp['P-Value'], method='fdr_bh')[1]

    best_penalty = xvp.iloc[0]['tt']
    mod = LogisticRegression(max_iter=100, penalty='l1', solver='liblinear', C=best_penalty)
    mod.fit(X, y)
    coef = mod.coef_.squeeze()
    
    # get variable importance
        # Compute standardized coefficients
    std_devs = X.std()
    standardized_coefficients = coef / std_devs.values
    
    variable_importance = pd.DataFrame({'Variable': list(X),
                                        'Raw Coefficient': coef,
                                        'Standardized Coefficient': standardized_coefficients,
                                        'Importance': abs(standardized_coefficients)
                                        }).sort_values(by='Importance', ascending=False)
        # Filter out zero-importance variables for plotting
    top_vars = variable_importance.head(top_n)
    if (top_vars['Importance'] == 0).any():   # if any of the top_n variables have importance value of 0
        top_vars = top_vars[top_vars['Importance'] > 0] # only work with vars that have none zero importance values
        
        # Plot variable importance. Put importance value on each bar
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_vars['Variable'], top_vars['Importance'], color='skyblue')
    for bar, st_coeff in zip(bars,  top_vars['Standardized Coefficient']):  # so on each bar it shows standardized coefficient
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{st_coeff:.4f}',
                 ha='center', va='bottom' if abs(st_coeff) > 0 else 'top')
    plt.xlabel('Variables')
    plt.ylabel('Importance (abs(Standardized Coefficient))')
    plt.title(f'Top {len(top_vars)} Variable Importance: {lbl}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_path = os.path.join(fld, f'Variable_Importance_{lbl.replace(" ", "_")}.png')
    plt.savefig(output_path)
    plt.show()
    kpZ = top_vars['Variable'].tolist()
    
    
    
    # Compute bootstrap CI for best cross-validated predictions
    ci_lower, ci_upper = 0.5, 0.5
    if best_y_true is not None and best_y_pred is not None:
        bootstrapped_aucs = []
        n_bootstraps = 100
        best_y_true = np.array(best_y_true)
        best_y_pred = np.array(best_y_pred)
        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(best_y_true)))
            y_true_bootstrap = best_y_true[indices]
            y_pred_bootstrap = best_y_pred[indices]
            if len(np.unique(y_true_bootstrap)) > 1:
                auc_val = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
                bootstrapped_aucs.append(auc_val)
        if len(bootstrapped_aucs) > 0:
            ci_lower = np.percentile(bootstrapped_aucs, 2.5)
            ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    # do single ROC curve on the entire cross-validated predictions (bestYHat)
    if bestYHat is not None:
        # bestYHat is a Series: the cross-validated predicted probabilities
        fpr, tpr, thresholds = roc_curve(y, bestYHat)  # 'y' is the entire outcome Series
    else:
        fpr, tpr, thresholds = [], [], []   # fallback if we have no CV preds
    
    
    # Build results dictionary
    train_results = {}
    train_results['model'] = mod
    train_results['kpZ'] = kpZ  # top 'top_n' variables to report based on var importance
    train_results['xpr_auc'] = xvp.iloc[0]['cstat']
    train_results['xpr_auc_ci_lower'] = ci_lower
    train_results['xpr_auc_ci_upper'] = ci_upper
    train_results['xpr_p_value'] = xvp.iloc[0]['P-Value']
    train_results['penalty_grid'] = xvp
    if returnYhat:
        train_results['yhat'] = bestYHat
    else:
        train_results['yhat'] = None
    
    train_results['xpr_fpr'] = fpr
    train_results['xpr_tpr'] = tpr
    train_results['xpr_thresholds'] = thresholds
    # train_results['variable_importance'] = variable_importance['Importance']  # all variable importance
    # train_results['raw_coefficients'] = variable_importance['Raw Coefficient']
    # train_results['standardized_coefficients'] = variable_importance['Standardized Coefficient']
    train_results['variable_importance'] = variable_importance  # the full variable importance DataFrame
    
    return train_results
    







def calculate_auc_ci2(y_true, y_scores, n_bootstraps=100):
    from sklearn.metrics import roc_auc_score
    import numpy as np

    # make sure inputs are Pandas Series
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_scores, np.ndarray):
        y_scores = pd.Series(y_scores)
    
    bootstrapped_auc = []
    for _ in range(n_bootstraps):
        # Bootstrap sampling
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue  # Skip if single class
        score = roc_auc_score(y_true.iloc[indices], y_scores.iloc[indices])
        bootstrapped_auc.append(score)
    
    if len(bootstrapped_auc) == 0:
        return np.nan, np.nan
    
    sorted_auc = np.sort(bootstrapped_auc)
    lower = sorted_auc[int(0.025 * len(sorted_auc))]
    upper = sorted_auc[int(0.975 * len(sorted_auc))]
    return lower, upper










def baseline_compare(y, yhat, baseline_df):
    """
    Replacement for 'euromacs_compare' that compares:
     1) Baseline-only model
     2) Combined model = Baseline + new cross-validated yhat

    Returns a dictionary with the same style of keys as your old euromacs_compare:
      - 'Baseline'
      - 'Combined'
      - 'Baseline_p-value-against-null'
      - 'Combined-p-value-againstBaseline'
      - 'Baseline_fpr', 'Baseline_tpr', etc.
    """
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Drop rows containing NAs
    Nna = baseline_df.isna().sum().sum()
    if Nna > 0:
        baseline_df = baseline_df.dropna()

    # Align indexes
    common_idx = y.index.intersection(yhat.index).intersection(baseline_df.index)
    df_temp = pd.DataFrame({
        'y': y.loc[common_idx].astype(float),
        'yhat_new': yhat.loc[common_idx].astype(float)
    })

    # If baseline_df has categorical columns, convert them to dummy variables
    baseline_enc = pd.get_dummies(baseline_df.loc[common_idx], drop_first=True)

    # Baseline-only model
    Xbase = sm.add_constant(baseline_enc)
    mod_base = sm.Logit(df_temp['y'], Xbase).fit(method='bfgs', disp=0)
    base_preds = mod_base.predict(Xbase)
    auc_base = roc_auc_score(df_temp['y'], base_preds)
    # 95% CI for baseline
    auc_ci_lower_base, auc_ci_upper_base = calculate_auc_ci2(df_temp['y'], base_preds)

    # Combined model (baseline + new analyte yhat)
    Xcomb = sm.add_constant(pd.concat([baseline_enc, df_temp[['yhat_new']]], axis=1))
    mod_comb = sm.Logit(df_temp['y'], Xcomb).fit(method='bfgs', disp=0)
    comb_preds = mod_comb.predict(Xcomb)
    auc_comb = roc_auc_score(df_temp['y'], comb_preds)
    # 95% CI for combined
    auc_ci_lower_comb, auc_ci_upper_comb = calculate_auc_ci2(df_temp['y'], comb_preds)

    # Baseline vs. null
    mod_null = sm.Logit(df_temp['y'], sm.add_constant(np.ones(len(df_temp)))).fit(disp=0)
    lr_base = 2 * (mod_base.llf - mod_null.llf)
    df_base = Xbase.shape[1] - 1
    pval_base_vs_null = stat.chi2.sf(lr_base, df_base)

    # Combined vs. baseline
    lr_comb = 2 * (mod_comb.llf - mod_base.llf)
    df_comb = Xcomb.shape[1] - Xbase.shape[1]
    pval_comb_vs_base = stat.chi2.sf(lr_comb, df_comb)

    # ROC data
    fpr_base, tpr_base, thr_base = roc_curve(df_temp['y'], base_preds)
    fpr_comb, tpr_comb, thr_comb = roc_curve(df_temp['y'], comb_preds)


    pf = {
        'Baseline': auc_base,
        'Baseline_95%CI_lower': auc_ci_lower_base,
        'Baseline_95%CI_upper': auc_ci_upper_base,
        'Baseline_p-value': pval_base_vs_null,   # vs. Null

        'Combined': auc_comb,
        'Combined_95%CI_lower': auc_ci_lower_comb,
        'Combined_95%CI_upper': auc_ci_upper_comb,
        'Combined-p-value-againstBaseline': pval_comb_vs_base,

        'Baseline_fpr': fpr_base,
        'Baseline_tpr': tpr_base,
        'Baseline_thresholds': thr_base,

        'Combined_fpr': fpr_comb,
        'Combined_tpr': tpr_comb,
        'Combined_thresholds': thr_comb
    }
    return pf







# baseline_cols = varListForAdvrsLvr 
baseline_columns = [#'Age_m',
                    'Baseline BMI (kg/m2)_m','baseline Fibrosis-4 (FIB-4) Score_m', 'baseline Albumin (g/dL)_m', 
                     'baseline Bilirubin (mg/dL)_m', 'baseline Creatinine (mg/dL)_m', 'baseline Hemoglobin A1C (%)_m']
baseline_cols = baseline_columns + ['Group']   # +[outcome]
# varListForAdvrsLvr = # 'Plate_Plate1', 
                     # 'Age_m',
                     # 'Baseline BMI (kg/m2)_m',
                     # 'baseline Albumin (g/dL)_m',
                     # 'baseline Bilirubin (mg/dL)_m',
                     # 'baseline Creatinine (mg/dL)_m',
                     # 'baseline Fibrosis-4 (FIB-4) Score_m',
                     # 'baseline Hemoglobin A1C (%)_m',
                     # 'Group',
                     # 'HCVGenotype_1.2',
                     # 'HCVGenotype_1.3',
                     # 'HCVGenotype_2.0',
                     # 'HCVGenotype_3.0',
                     # 'HCVGenotype_4.0',
                     # 'HCVGenotype_5.0'
Dem = dem[baseline_cols]   # does NOT contain 'AGE' term









def riskModel(outcome,xpr, lbl, PCA=False, PCA_n_component=0.9):
    cohort = list(set(Dem.loc[~Dem[outcome].isna()].index.values) & set(xpr.index.values)) 
    analyteList = [v for b,v in zip((xpr.loc[cohort].isna().sum()==0),list(xpr)) if b]   ### Use only the analytes with no missing data
    
    
    if PCA:
        logging.info(f"Running PCA-based model for {lbl} (n_components={PCA_n_component})")
        ### PCA
            # PCA to explain 90% of variance
        scaler = StandardScaler()
        pca_90 = PCA(n_components=PCA_n_component)  # default PCA_n_component=0.9
        pipeline_90 = make_pipeline(scaler, pca_90)
        data_transformed_90 = pipeline_90.fit_transform(xpr.loc[cohort, analyteList])
            # Convert the PCA-transformed data back into a DataFrame for prediction
        pc_columns_90 = [f'PC{i+1}' for i in range(data_transformed_90.shape[1])]
        data_pca_90 = pd.DataFrame(data_transformed_90, index=xpr.loc[cohort, analyteList].index, columns=pc_columns_90)
    
        ### Extract PCA Loadings
        pca_loadings = pca_90.components_.T  # Transpose to get loadings
        biomarker_names = analyteList   
        pca_loadings_df = pd.DataFrame(pca_loadings, index=biomarker_names, columns=pc_columns_90)
        ### Will save PC loadings later in code

        train_results = selectAndTrain_original_V3(data_pca_90, clin.loc[cohort, outcome], lbl=lbl, verbose=True, leaveOneOut=True,  top_n=10)
    
    else:
        logging.info(f"Running NON-PCA model for {lbl}"
        train_results = selectAndTrain_original_V3(xpr.loc[cohort, analyteList], Dem.loc[cohort, outcome], lbl=lbl, returnYhat=True, verbose=True, leaveOneOut=True, top_n=10)
        pca_loadings_df = None
    
    
    
        # Convert yhat to a pandas Series with the correct index
    # Get the performance metrics from euromacs_compare 
    #pf=euromacs_compare(clin.loc[cohort,outcome],train_results['yhat'],clin.loc[cohort,'pre EUROMACS'])
    if train_results['model'] is not None:
        # Compare with Baseline
        pf = baseline_compare(y=Dem.loc[cohort, outcome], yhat=train_results['yhat'], baseline_df=dem.loc[cohort, baseline_columns]   )
    else:
        logging.warning(f"{lbl}: Model training failed. Skipping Baseline comparison.")
        pf = {}
        
    
    final_results = {
        'perf_metrics': {'Analytes AUC': train_results['xpr_auc'],   # 'c-statistic'
                         'Analytes 95% CI lower': train_results['xpr_auc_ci_lower'],
                         'Analytes 95% CI upper': train_results['xpr_auc_ci_upper'],
                         'Analytes P-value': train_results['xpr_p_value'],

                         'Baseline AUC': pf.get('Baseline', np.nan),
                         'Baseline 95%CI lower': pf.get('Baseline_95%CI_lower', np.nan) ,
                         'Baseline 95%CI upper': pf.get('Baseline_95%CI_upper', np.nan),
                         'Baseline p-value vs. Null': pf.get('Baseline_p-value', np.nan),

                         'Combined AUC': pf.get('Combined', np.nan),
                         'Combined 95%CI lower': pf.get('Combined_95%CI_lower', np.nan) ,
                         'Combined 95%CI upper': pf.get('Combined_95%CI_upper', np.nan),
                         'Combined p-value vs. Baseline': pf.get('Combined-p-value-againstBaseline', np.nan),

                         'Predictors': ', '.join(train_results['kpZ'])
        },
        
        'roc_data': {'Baseline_fpr': pf.get('Baseline_fpr', []),
                     'Baseline_tpr': pf.get('Baseline_tpr', []),
                     'Combined_fpr': pf.get('Combined_fpr', []),
                     'Combined_tpr': pf.get('Combined_tpr', [])
        },
        
        'trained_model': train_results,  # the dictionary from selectAndTrain_original_V3
        'pca_loadings': pca_loadings_df  # None if PCA=False
    }
    
        
    logging.info(f"{lbl}: Model training and evaluation complete.")
    return final_results









#### Plot ROC curves
def plot_roc_curves(roc_data_collection, color_map, output_filename, results, performanceTable):
    plt.figure(figsize=(10, 8))
    legend_labels = []
    
    # Plot each ROC curve
    for i, (label, roc_info) in enumerate(roc_data_collection.items()):
        if label in color_map:
            # Extract the appropriate metrics based on the label
            if "AdverseLiverEvents Proteomics_Baseline" in label:
                fpr_key = 'Baseline_fpr'
                tpr_key = 'Baseline_tpr'
                metric_key = 'Baseline AUC'
            elif "AdverseLiverEvents Proteomics_Combined" in label:
                fpr_key = 'Combined_fpr'
                tpr_key = 'Combined_tpr'
                metric_key = 'Combined AUC'
            elif "AdverseLiverEvents Metabolomics_Combined" in label:
                fpr_key = 'Combined_fpr'
                tpr_key = 'Combined_tpr'
                metric_key = 'Combined AUC'
            elif "AdverseLiverEvents MetabolomicProteomic_Combined" in label:
                fpr_key = 'Combined_fpr'
                tpr_key = 'Combined_tpr'
                metric_key = 'Combined AUC'
            else:
                print(f"Invalid label '{label}'")
                continue
            
            # Use the metric from performanceTable if available, otherwise use the one from results
            if "AdverseLiverEvents Proteomics_Baseline" in label:
                auc_score = performanceTable.loc['AdverseLiverEvents Proteomics', metric_key]
            elif "AdverseLiverEvents Proteomics_Combined" in label:
                auc_score = performanceTable.loc['AdverseLiverEvents Proteomics', metric_key]
            elif "AdverseLiverEvents Metabolomics_Combined" in label:
                auc_score = performanceTable.loc['AdverseLiverEvents Metabolomics', metric_key]
            else:
                auc_score = performanceTable.loc['AdverseLiverEvents MetabolomicProteomic', metric_key]            
            legend_label = f"{label}. AUC Score = {auc_score:.4f}"
            
            # Plot the ROC curve with the specified color and legend label
            if i == 0:  # Check if it's the first curve (proteomics EUROMACS-only in dashed line)
                plt.plot(roc_info[fpr_key], roc_info[tpr_key], label=legend_label, color=color_map[label], linestyle='--')
            else:
                plt.plot(roc_info[fpr_key], roc_info[tpr_key], label=legend_label, color=color_map[label])
            
            legend_labels.append(legend_label)  # Add to legend labels for additional information
        else:
            print(f"Label '{label}' not found in color_map.")
    if not legend_labels:
        print("No valid legend labels found.")
    else:
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.savefig(fld + output_filename + '.png', dpi=300)
        plt.show()

                    
### The full cohort -------------------------------------------------------------------------------------------------------
# expression levels matrices
    # only proteomics
proteomics = xpr
    # only metabolomics
metabolomics = lab_xpr   # so with the aggregate vars

#     # if wanted to get rid of aggregate variables (such as 'Sum of ..." "ratio of ...") == empty 'Class' row in Q500 and Bile_Acids_Final
# metabolomics = lab_xpr.loc[:, ~lab_xpr.columns.isin(empty_class_analytes)]

    # proteomics and metablomics
common_indices = proteomics.index.intersection(metabolomics.index)
proteomicsAndMetabolomics = pd.concat([proteomics.loc[common_indices], metabolomics.loc[common_indices]], axis=1)

models={"AdverseLiverEvents Proteomics": proteomics,
        "AdverseLiverEvents Metabolomics": metabolomics,
        "AdverseLiverEvents MetabolomicProteomic": proteomicsAndMetabolomics   #,
        # # until later if they ask for it
        # "RV Delta Proteomics":deltaX_lvad, 
        # "RV Delta Metabolomics":mdeltaX_lvad, 
        # "RV Delta MetabolomicProteomic":metaboloProteo_delta_lvad  
        }     



pca_loadings_dfs = {}
perf_rows = []
roc_data_collection = {}
results_full = {}  # store the full results, including pipeline/model

# initialize prediction columns in clin for each model, keep these columns for future use
for label in models.keys():
    Dem[f'yhat_full_{label}'] = np.nan 

for label, xpr_data in models.items():
    logging.info(f"Starting model: {label}")
    
    results = riskModel("Group", xpr_data, label)
    #results = riskModel("Group", xpr_data, labe, PCA=True, PCA_n_component=0.9)
    
    
    
    
    
        #save the entire result object for later use
    results_full[label] = results
        # for the existing performance table on full cohort
    if results['perf_metrics']:
        # save PCA loadings
        pca_loadings_dfs[label] = results['pca_loadings']
        # append performance metrics
        perf_row = pd.Series(results['perf_metrics'], name=label)
        perf_rows.append(perf_row)
        # store ROC data
        roc_data_collection[label] = results['roc_data']
        # store cross-validated predictions in the clin dataframe
        yhat_full = results['trained_model']['yhat']
        Dem.loc[yhat_full.index, f'yhat_full_{label}'] = yhat_full
        # save the variable importance info
        variable_importance = results['trained_model']['variable_importance']
        #variable_importance.to_excel(f'Variable_Importance_{label.replace(" ", "_")}.xlsx', index=False)
        excel_path = os.path.join(fld, 'Variable_Importance_AllModels.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a' if os.path.exists(excel_path) else 'w') as writer:
            variable_importance.to_excel(writer, sheet_name=label, index=False)
    else:
        logging.warning(f"{label}: No performance metrics available (single-class outcome or training failed).")
    
    
# Save PCA loadings
pca_excel_filepath = fld + 'PCA_Loadings_All_RV_Models.xlsx'   # obtained by running the models by hand, one by one
with pd.ExcelWriter(pca_excel_filepath) as writer:
    for label, pca_loadings_df in pca_loadings_dfs.items():
        pca_loadings_df.to_excel(writer, sheet_name=label)
logging.info(f"PCA loadings saved to {pca_excel_filepath}")

# save the performance results
if perf_rows:
    perf = pd.concat(perf_rows, axis=1)
    perfT = perf.T
    perf_excel_filepath = fld + 'HF_ModelPerformance_AdverseLiverEvents.xlsx'
    perfT.to_excel(perf_excel_filepath)
    logging.info(f"Full cohort performance saved to {perf_excel_filepath}")
else:
    logging.warning("No performance metrics to save for full cohort.")


# ROC curves
color_map = dict(zip(["AdverseLiverEvents Proteomics_Baseline",
                      "AdverseLiverEvents Proteomics_Combined",
                      "AdverseLiverEvents Metabolomics_Combined",
                      "AdverseLiverEvents MetabolomicProteomic_Combined"],
                     vs_palette()[:4]))
# Define subsets of ROC data and corresponding color mappings for each figure
subset_1_roc_data = {'AdverseLiverEvents Proteomics_Baseline': roc_data_collection['AdverseLiverEvents Proteomics']}
subset_1_color_map = {'AdverseLiverEvents Proteomics_Baseline': color_map['AdverseLiverEvents Proteomics_Baseline']}

subset_2_roc_data = {'AdverseLiverEvents Proteomics_Baseline': roc_data_collection['AdverseLiverEvents Proteomics'],
                      'AdverseLiverEvents Proteomics_Combined': roc_data_collection['AdverseLiverEvents Proteomics']}
subset_2_color_map = color_map.copy()

subset_3_roc_data = {'AdverseLiverEvents Proteomics_Baseline': roc_data_collection['AdverseLiverEvents Proteomics'],
                      'AdverseLiverEvents Proteomics_Combined': roc_data_collection['AdverseLiverEvents Proteomics'],
                      'AdverseLiverEvents Metabolomics_Combined': roc_data_collection['AdverseLiverEvents Metabolomics']}
subset_3_color_map = color_map.copy()

subset_4_roc_data = {'AdverseLiverEvents Proteomics_Baseline': roc_data_collection['AdverseLiverEvents Proteomics'],
                     'AdverseLiverEvents Proteomics_Combined': roc_data_collection['AdverseLiverEvents Proteomics'],
                     'AdverseLiverEvents Metabolomics_Combined': roc_data_collection['AdverseLiverEvents Metabolomics'],
                     'AdverseLiverEvents MetabolomicProteomic_Combined': roc_data_collection['AdverseLiverEvents MetabolomicProteomic']}
subset_4_color_map = color_map.copy()

plot_roc_curves(subset_1_roc_data, subset_1_color_map, 'ROC_figure_1, full cohort', results, perfT)
plot_roc_curves(subset_2_roc_data, subset_2_color_map, 'ROC_figure_2, full cohort', results, perfT)
plot_roc_curves(subset_3_roc_data, subset_3_color_map, 'ROC_figure_3, full cohort', results, perfT)
plot_roc_curves(subset_4_roc_data, subset_4_color_map, 'ROC_figure_4, full cohort', results, perfT)





