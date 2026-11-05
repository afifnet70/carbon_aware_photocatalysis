#!/usr/bin/env python
import argparse, json, os, platform, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import joblib

from src.data import load_dataset, split_data
from src.models import model_specs, TargetScaledRegressor
from src.features import FeatureBuilder
from src.config import CARBON

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/glfo_hydrogen.csv")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--quick",action="store_true")
    ap.add_argument("--output",default="results")
    ap.add_argument("--cores",type=int,default=2)
    ap.add_argument("--watts-per-core",type=float,default=12.0)
    ap.add_argument("--pue",type=float,default=1.0)
    ap.add_argument("--carbon-intensity",type=float,default=475.0,
                    help="g CO2e per kWh")
    args=ap.parse_args()
    out=Path(args.output); (out/"figures").mkdir(parents=True,exist_ok=True)
    carbon=CARBON
    carbon.busy_cores=args.cores
    carbon.watts_per_core=args.watts_per_core
    carbon.pue=args.pue
    carbon.carbon_intensity_g_per_kwh=args.carbon_intensity

    df=load_dataset(args.data)
    X_train,X_test,y_train,y_test=split_data(df,args.seed)
    print(f"Dataset: {len(df)} rows | train={len(X_train)} | hold-out={len(X_test)}")
    print(df.describe().to_string())

    cv=KFold(n_splits=3,shuffle=True,random_state=args.seed)
    rows=[]; search_rows=[]; fitted={}
    specs=model_specs(args.seed,args.quick)
    for spec in specs:
        best=None; best_score=np.inf; best_set=None; total_search_seconds=0.0
        for feature_set in spec["feature_sets"]:
            pipe=Pipeline([
                ("features",FeatureBuilder(feature_set=feature_set,random_state=args.seed)),
                ("scale",__import__("sklearn").preprocessing.MinMaxScaler()),
                ("model",clone(spec["estimator"]))
            ])
            if spec["target_scale"]:
                estimator=TargetScaledRegressor(pipe)
                # Parameter names are nested under estimator.estimator.
                params={"estimator__"+k:v for k,v in spec["params"].items()}
            else:
                estimator=pipe
                params=spec["params"]
            n_iter=1 if not params else spec["n_iter"]
            search=RandomizedSearchCV(
                estimator=estimator,param_distributions=params,n_iter=n_iter,
                scoring="neg_root_mean_squared_error",cv=cv,random_state=args.seed,
                n_jobs=1,refit=True,verbose=0,error_score="raise"
            )
            t0=time.perf_counter()
            search.fit(X_train,y_train.to_numpy())
            elapsed=time.perf_counter()-t0
            total_search_seconds+=elapsed
            if -search.best_score_<best_score:
                best_score=-search.best_score_; best=search.best_estimator_
                best_set=feature_set; best_params=search.best_params_
            search_rows.append({"model":spec["name"],"feature_set":feature_set,
                                "search_wall_seconds":elapsed,"draws":n_iter,
                                "best_cv_rmse":-search.best_score_})
            print(f"{spec['name']:26s} {feature_set:9s} CV RMSE={-search.best_score_:.6f}")
        # Final selected model fit: median of three independent fits.
        fit_times=[]; preds=[]
        for rep in range(3):
            candidate=clone(best)
            wall0=time.perf_counter(); cpu0=time.process_time(); candidate.fit(X_train,y_train.to_numpy())
            fit_times.append(max(time.perf_counter()-wall0, time.process_time()-cpu0))
            inf_wall=time.perf_counter(); inf_cpu=time.process_time(); pred_now=candidate.predict(X_test)
            inference_seconds=max(time.perf_counter()-inf_wall,time.process_time()-inf_cpu)
            preds.append(pred_now)
            if rep==2: final_model=candidate
        fit_seconds=float(np.median(fit_times))
        y_pred=np.mean(np.vstack(preds),axis=0)
        # The manuscript reports metrics for one final selected fit; use the
        # third seeded fit for prediction and record the median fit time.
        y_pred=preds[-1]
        rmse=float(np.sqrt(mean_squared_error(y_test,y_pred)))
        mae=float(mean_absolute_error(y_test,y_pred))
        r2=float(r2_score(y_test,y_pred))
        fit_wh=carbon.energy_wh(fit_seconds)
        fit_co2=carbon.emissions_g(fit_wh)
        # Tuning wall time is treated as two busy cores (upper bound).
        tuning_wh=carbon.energy_wh(total_search_seconds)
        tuning_co2=carbon.emissions_g(tuning_wh)
        row={"model":spec["name"],"family":spec["family"],"feature_set":best_set,
             "cv_rmse":best_score,"R2":r2,"RMSE":rmse,"MAE":mae,
             "fit_cpu_seconds_median":fit_seconds,"fit_energy_Wh":fit_wh,
             "fit_CO2e_g":fit_co2,"inference_cpu_seconds":inference_seconds,
             "inference_energy_Wh":carbon.energy_wh(inference_seconds),
             "inference_CO2e_g":carbon.emissions_g(carbon.energy_wh(inference_seconds)),
             "search_wall_seconds":total_search_seconds,
             "search_energy_Wh_upper_bound":tuning_wh,
             "search_CO2e_g_upper_bound":tuning_co2,
             "best_params":json.dumps(best_params,default=str)}
        rows.append(row); fitted[spec["name"]]=(final_model,spec)
        print(f"  -> holdout R2={r2:.6f} RMSE={rmse:.6f} fit={fit_seconds:.3f}s")
        if spec["name"]=="ANN-3 (tanh, L-BFGS)":
            joblib.dump(final_model,out/"ann3_model.joblib")
            pd.DataFrame({"measured":y_test.to_numpy(),"predicted":y_pred,
                          "residual":y_test.to_numpy()-y_pred}).to_csv(
                out/"holdout_predictions.csv",index=False)
            from src.explain import run_explanations
            run_explanations(final_model,X_train,X_test,y_test,out/"figures"/"explainability",args.seed)

    result=pd.DataFrame(rows).sort_values("R2",ascending=False)
    result.to_csv(out/"benchmark_results.csv",index=False)
    search_df=pd.DataFrame(search_rows)
    search_df.to_csv(out/"search_cost.csv",index=False)

    # Overall tuning upper bound includes the measured search operations.
    total_search_wh=float(result["search_energy_Wh_upper_bound"].sum())
    total_search_co2=float(result["search_CO2e_g_upper_bound"].sum())
    metadata={
      "seed":args.seed,"split":"random 80/20 hold-out; 3-fold shuffled KFold",
      "n_rows":len(df),"n_train":len(X_train),"n_holdout":len(X_test),
      "python":platform.python_version(),"scikit_learn":sklearn.__version__,
      "platform":platform.platform(),"quick_mode":args.quick,
      "carbon_assumptions":{"watts_per_busy_core":args.watts_per_core,
        "busy_cores":args.cores,"PUE":args.pue,
        "carbon_intensity_g_CO2e_per_kWh":args.carbon_intensity,
        "memory_power_included":False,"accelerator_used":False},
      "search_energy_Wh_upper_bound":total_search_wh,
      "search_CO2e_g_upper_bound":total_search_co2,
      "reference_experiment_Wh":carbon.reference_experiment_wh,
      "note":"The manuscript does not specify the numeric split seed; default implementation seed is 42."
    }
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    print("\\nBenchmark complete.")
    print(result[["model","feature_set","R2","RMSE","MAE","fit_cpu_seconds_median",
                  "fit_energy_Wh","fit_CO2e_g"]].to_string(index=False))
    print(f"Search energy upper bound: {total_search_wh:.3f} Wh")
    print(f"Search emissions upper bound: {total_search_co2:.3f} g CO2e")

    # Benchmark plot
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7))
    for fam,g in result.groupby("family"):
        ax.scatter(g["fit_cpu_seconds_median"],g["RMSE"],label=fam,s=55)
        for _,r in g.iterrows():
            ax.annotate(r["model"].replace(" (tanh, L-BFGS)",""),
                        (r["fit_cpu_seconds_median"],r["RMSE"]),fontsize=7,xytext=(3,3),
                        textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Final-fit CPU time (s)"); ax.set_ylabel("Hold-out RMSE")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/"figures"/"cost_accuracy.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(result["model"],result["fit_energy_Wh"])
    ax.set_ylabel("Final-fit energy (Wh)"); ax.tick_params(axis="x",rotation=80)
    ax.grid(axis="y",alpha=.2); fig.tight_layout()
    fig.savefig(out/"figures"/"final_fit_energy.png",dpi=300); plt.close(fig)

if __name__=="__main__":
    main()
