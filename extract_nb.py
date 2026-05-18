import json
nb = json.load(open('notebooks/semantic-segmentattion-unet.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if any(kw in src.lower() for kw in ['compile', 'adam', 'fit(', 'callback', 'earlystop', 'reduce', 'epoch', 'batch_size', 'learning']):
            print(f'=== Cell {i} ===')
            print(src[:2000])
            print()
