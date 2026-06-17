import requests, json, threading, queue, time, sys

BASE = 'http://127.0.0.1:5001/api'

event_queue = queue.Queue()
sse_done = threading.Event()

payload = {
    'pde_type': 'heat',
    'equation_latex': r'u_t = \alpha u_{xx}',
    'ic_expression': 'sin(pi*x)',
    'bc_left_expression': '0',
    'bc_right_expression': '0',
    'domain': {'x_min': 0, 'x_max': 1, 't_min': 0, 't_max': 1},
    'params': {'alpha': 0.01},
    'layers': [2, 32, 32, 1],
    'n_collocation': 2000,
    'n_initial': 100,
    'n_boundary': 100,
    'epochs': 100,
    'learning_rate': 0.001,
    'name': 'api_integration_test',
    'use_fourier': True,
    'use_adaptive_activation': True,
    'use_hard_constraint': True,
    'use_mc_dropout': True,
    'fourier_bands': [0.1, 1.0, 10.0],
    'fourier_freqs': 16,
    'dropout_rate': 0.05,
    'mc_samples': 8,
    'log_interval': 20,
}

def stream_reader():
    try:
        r = requests.get(f'{BASE}/stream', stream=True, timeout=180)
        current_event = 'message'
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith('event:'):
                current_event = line[6:].strip()
            elif line.startswith('data:'):
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                    event_queue.put((current_event, data))
                    if current_event in ('complete', 'error'):
                        sse_done.set()
                        return
                except Exception as e:
                    print(f'[stream parse err] {e}, data="{data_str[:80]}"')
    except Exception as e:
        print(f'Stream error: {e}')
        sse_done.set()

t = threading.Thread(target=stream_reader, daemon=True)
t.start()
time.sleep(1.0)

print('=== POST /api/solve ===')
resp = requests.post(f'{BASE}/solve', json=payload, timeout=20)
print('Status:', resp.status_code)
print('Response:', json.dumps(resp.json(), indent=2, ensure_ascii=False))

print()
print('=== 等待SSE事件 ===')
t0 = time.time()
event_counts = {}
progress_samples = []
while not sse_done.wait(0.5):
    if time.time() - t0 > 120:
        print('TIMEOUT 2min')
        break
    while not event_queue.empty():
        etype, edata = event_queue.get()
        event_counts[etype] = event_counts.get(etype, 0) + 1
        if etype == 'connected':
            print(f'[connected] client_id={edata.get("client_id")}')
        elif etype == 'progress':
            if len(progress_samples) < 3:
                progress_samples.append(edata)
            if event_counts[etype] == 1 or event_counts[etype] % 3 == 0:
                print(f"[progress #{event_counts[etype]}] epoch={edata.get('epoch')} loss={edata.get('loss'):.4e}")
        elif etype == 'complete':
            print(f'[complete] id={edata.get("id")}, final_loss={edata.get("final_loss"):.4e}')
            gd = edata.get('grid_data') or {}
            print(f'  grid x:{len(gd.get("x",[]))}  t:{len(gd.get("t",[]))}')
            print(f'  has_uncertainty: {gd.get("has_uncertainty")}')
            if gd.get('u_uncertainty'):
                uflat = [v for row in gd['u_uncertainty'] for v in row]
                print(f'  uncertainty max={max(uflat):.3e}  mean={sum(uflat)/len(uflat):.3e}')
            print(f'  history points returned: {len(edata.get("training_history", []))}')
        elif etype == 'error':
            print(f'[error] {edata}')
        elif etype == 'heartbeat':
            pass
        else:
            print(f'[{etype}] (count={event_counts[etype]})')

print()
print(f'Total elapsed: {time.time()-t0:.1f}s')
print('Event counts:', event_counts)

print()
print('=== 验证 /api/history ===')
h = requests.get(f'{BASE}/history', timeout=10).json()
print(f'History records: {len(h)}')
if h:
    latest = h[-1]
    print(f'Latest: id={latest["id"]} name={latest["name"]} final_loss={latest["final_loss"]}')
    print(f'  pde_type={latest["pde_type"]}  epochs={latest["epochs"]}')

print()
print('ALL INTEGRATION TESTS PASSED' if 'complete' in event_counts else 'FAILED: no complete event')
