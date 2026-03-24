content = open('/tmp/postprocess_plots.py').read()
old = 'float(params["R0"]) if np.isfinite(params.get("R0", np.nan))'
new = '(float(params["R0"]) if str(params.get("R0","")).strip() not in ("","nan","None") else 1.0) if False else (float(params["R0"]) if (lambda v: isinstance(v,(int,float)) and v==v and v!=float("inf"))(params.get("R0",float("nan"))) '
# simpler approach: just replace problematic call
old2 = 'R0_init=float(params["R0"]) if np.isfinite(params.get("R0", np.nan)) else 1.0,'
new2 = 'R0_init=(float(params["R0"]) if isinstance(params.get("R0"), (int,float)) and not (params.get("R0") != params.get("R0")) else 1.0),'
fixed = content.replace(old2, new2)
open('/tmp/postprocess_plots.py', 'w').write(fixed)
print('patched' if old2 not in fixed else 'NOT patched - line not found')
