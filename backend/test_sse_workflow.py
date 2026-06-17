import requests, json, threading, queue, time

BASE = 'http://127.0.0.1:5001/api'

event_queue = queue.Queue()
sse_done = threading.Event()

payload = {
    'pde_type': 'heat',
    'equation_latex': r'\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}',
    'ic_expression': 'sin(pi*x) + 0.1*sin(10*pi*x)',
    'bc_left_expression': '0',
    'bc_right_expression': '0',
    'domain': {'x_min': 0, 'x_max': 1, 't_min': 0, 't_max': 1},
    'params': {'alpha': 0.01},
    'layers': [2, 64, 64, 1],
    'n_collocation': 5000,
    'n_initial': 200,
    'n_boundary': 200,
    'epochs': 600,
    'learning_rate': 0.001,
    'name': 'test_integration',
    'use_fourier': True,
    'use_adaptive_activation': True,
    'use_hard_constraint': True,
    'use_mc_dropout': True,
    'fourier_bands': [0.01, 0.1, 1, 10],
    'fourier_freqs': 32,
    'dropout_rate': 0.05,
    'mc_samples': 15,
    'log_interval': 50,
}

def stream_reader():
    try:
        r = requests.get(f'{BASE}/stream', stream=True, timeout=120)
        buf = ''
        for line in r.iter_lines(decode_unicode=True):
            if line is None: continue
            if line.startswith('event:'):
                event_type = line[6:].strip()
            elif line.startswith('data:'):
                data_str = line[5:].strip()
                if not data_str: continue
                try:
                    data = json.loads(data_str)
                    event_queue.put((event_type, data))
                    if event_type == 'complete' or event_type == 'error':
                        sse_done.set()
                        return
                except Exception as e:
                    print(f'parse err: {e}, data: {data_str}')
    except Exception as e:
        print(f'Stream error: {e}')
        sse_done.set()

t = threading.Thread(target=stream_reader, daemon=True)
t.start()
time.sleep(1.2)

start = time.time()
print('POST /solve ...')
resp = requests.post(f'{BASE}/solve', json=payload, timeout=30)
print('start status:', resp.status_code, resp.json())

counts = {'progress': 0, 'complete': 0, 'error': 0, 'connected': 0}
progress_samples = []
while not sse_done.wait(0.2):
    if time.time() - start > 600:
        print('TIMEOUT 10min, abort')
        break
    while not event_queue.empty():
        etype, edata = event_queue.get()
        counts[etype] = counts.get(etype, 0) + 1
        if etype == 'connected':
            print(f'[connected] client={edata.get("client_id")}')
        elif etype == 'progress':
            if len(progress_samples) < 4:
                progress_samples.append(edata)
            if counts['progress'] % 2 == 0:
                print(f"[progress #{counts['progress']}] epoch={edata.get('epoch')} loss={edata.get('loss'):.4e}")
        elif etype == 'complete':
            print(f'[complete] id={edata.get("id")}, final_loss={edata.get("final_loss"):.4e}')
            gd = edata.get('grid_data') or {}
            print(f'  grid x:{len(gd.get("x",[]))}  t:{len(gd.get("t",[]))}')
            print(f'  has_uncertainty: {gd.get("has_uncertainty")}')
            if gd.get('u_uncertainty'):
                uflat = gd['u_uncertainty']
                flat = []
                for row in uflat: flat.extend(row)
                print(f'  uncertainty max={max(flat):.3e} mean={sum(flat)/len(flat):.3e}')
            print(f'  history points: {len(edata.get("training_history", []))}')
        elif etype == 'error':
            print(f'[error] {edata}')

print()
print('Final event counts:', counts)
print(f'First few progress samples: {json.dumps(progress_samples, ensure_ascii=False)}')
