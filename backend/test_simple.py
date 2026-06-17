import requests, json, time

BASE = 'http://127.0.0.1:5001/api'

payload = {
    'pde_type': 'heat',
    'equation_latex': r'\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}',
    'ic_expression': 'sin(pi*x)',
    'bc_left_expression': '0',
    'bc_right_expression': '0',
    'domain': {'x_min': 0, 'x_max': 1, 't_min': 0, 't_max': 1},
    'params': {'alpha': 0.01},
    'layers': [2, 64, 64, 1],
    'n_collocation': 5000,
    'n_initial': 200,
    'n_boundary': 200,
    'epochs': 200,
    'learning_rate': 0.001,
    'name': 'test_history_check',
    'use_fourier': True,
    'use_adaptive_activation': True,
    'use_hard_constraint': True,
    'use_mc_dropout': True,
    'fourier_bands': [0.01, 0.1, 1, 10],
    'fourier_freqs': 32,
    'dropout_rate': 0.05,
    'mc_samples': 10,
    'log_interval': 50,
}

t0 = time.time()
print('POST /solve...')
r = requests.post(f'{BASE}/solve', json=payload, timeout=20)
print('solve:', r.status_code, r.json())
task_id = r.json()['task_id']

for i in range(60):
    time.sleep(3)
    h = requests.get(f'{BASE}/history', timeout=10).json()
    print(f'[{i+1}/60, elapsed {time.time()-t0:.1f}s] history records:', len(h))
    if h:
        latest = h[-1]
        print(f'  latest: id={latest["id"]} name={latest["name"]} final_loss={latest["final_loss"]} created_at={latest["created_at"]}')
        if latest['final_loss'] is not None and time.time() - t0 > 5:
            rid = latest['id']
            print(f'  fetching predict for record {rid}...')
            pr = requests.post(f'{BASE}/predict/{rid}',
                               json={'nx': 50, 'nt': 50, 'mc_samples': 10, 'compute_uncertainty': True},
                               timeout=60)
            print(f'  predict status: {pr.status_code}, keys:', list(pr.json().keys()))
            gd = pr.json()
            print(f'  has_uncertainty: {gd.get("has_uncertainty")}')
            if gd.get('u_uncertainty'):
                flat = [v for row in gd['u_uncertainty'] for v in row]
                print(f'  u_uncertainty max={max(flat):.3e} mean={sum(flat)/len(flat):.3e}')
            break
    if time.time() - t0 > 180:
        print('TIMEOUT')
        break
