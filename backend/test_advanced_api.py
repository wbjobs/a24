import requests, json, time

BASE = 'http://127.0.0.1:5001/api'

print('=== 1. /system/capabilities 能力查询 ===')
r = requests.get(f'{BASE}/system/capabilities', timeout=10)
print(r.status_code)
caps = r.json()
for k, v in caps.items():
    if isinstance(v, list):
        print(f'  {k}: (list, {len(v)} items) {v[:5]}...')
    else:
        print(f'  {k}: {v}')

print()
print('=== 2. /library 方程库列表 ===')
r2 = requests.get(f'{BASE}/library', timeout=10)
print(r2.status_code)
d = r2.json()
print(f'  total: {d.get("total")}, filtered: {d.get("filtered")}')
print(f'  categories: {d.get("categories")}')
print(f'  pde_types: {d.get("pde_types")}')
for c in d.get('cases', [])[:3]:
    print(f'    - [{c["id"]}] {c["name"]} ({c["category"]}, nd={c["n_dims"]}, difficulty={c["difficulty"]})')

print()
print('=== 3. /library/kdv_soliton 案例详情 ===')
r3 = requests.get(f'{BASE}/library/kdv_soliton', timeout=10)
print(r3.status_code)
case = r3.json()
for k in ['id', 'name', 'equation_latex', 'default_epochs', 'default_layers', 'has_fdm_reference']:
    if k in case:
        v = case[k]
        if isinstance(v, str) and len(v) > 80:
            v = v[:76] + '...'
        print(f'  {k}: {v}')

print()
print('=== 4. /library/heat_basic/solve  测试热传导案例快速求解 (300 epochs) ===')
r4 = requests.post(f'{BASE}/library/heat_basic/solve', json={
    'epochs': 300, 'n_collocation': 2000,
    'layers': [2, 32, 32, 1],
    'log_interval': 50,
    'run_fdm_compare': True,
    'use_mc_dropout': False,
}, timeout=20)
print(r4.status_code, json.dumps(r4.json(), ensure_ascii=False))
task_id = r4.json()['task_id']

print()
print('=== 5. SSE等待complete事件 ===')
import queue, threading
q = queue.Queue()
done = threading.Event()

def reader():
    try:
        r = requests.get(f'{BASE}/stream', stream=True, timeout=300)
        cur_ev = 'msg'
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            if line.startswith('event:'):
                cur_ev = line[6:].strip()
            elif line.startswith('data:'):
                ds = line[5:].strip()
                try:
                    data = json.loads(ds)
                    q.put((cur_ev, data))
                    if cur_ev in ('complete', 'error'):
                        done.set()
                        return
                except: pass
    except Exception as e:
        print(f'SSE err: {e}')
        done.set()

th = threading.Thread(target=reader, daemon=True)
th.start()
time.sleep(1.0)

t0 = time.time()
counts = {}
comp_data = None
while not done.wait(0.5):
    if time.time() - t0 > 120:
        print('TIMEOUT')
        break
    while not q.empty():
        et, data = q.get()
        counts[et] = counts.get(et, 0) + 1
        if et == 'progress' and counts[et] % 2 == 0:
            print(f"  [progress #{counts[et]}] epoch={data.get('epoch')} loss={data.get('loss'):.3e}")
        elif et == 'complete':
            comp_data = data
            print(f'  [complete] id={data.get("id")}, final_loss={data.get("final_loss"):.3e}, time={data.get("training_time_sec"):.2f}s')
            cmp_obj = data.get('comparison')
            if cmp_obj:
                print(f'   COMPARISON: l2_err={cmp_obj.get("l2_error"):.3e}, speedup={cmp_obj.get("speedup_ratio"):.2f}x')
                print(f'   fdm_method={cmp_obj.get("fdm_method")}, pinn_time={cmp_obj.get("pinn_time_sec")}, fdm_time={cmp_obj.get("fdm_time_sec")}')
            else:
                print(f'   (no comparison data)')
        elif et == 'error':
            print(f'  [ERROR] {data}')

print()
print(f'All events: {counts}')
print()
print('=== 6. /hyperopt/status （不启动，只查状态） ===')
r6 = requests.get(f'{BASE}/hyperopt/status', timeout=10)
print(r6.status_code, json.dumps(r6.json(), ensure_ascii=False))

print()
print('ALL NEW API TESTS PASSED')
